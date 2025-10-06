import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import numpy as np
from ultralytics import YOLO
import yaml

class TrainingManager:
    def __init__(self, base_dir: str = "training_data"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.runs_dir = Path("runs")
        self.results_dir = self.base_dir / "results"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get all training session history"""
        history = []

        if not self.runs_dir.exists():
            return history

        detect_dir = self.runs_dir / "detect"
        if not detect_dir.exists():
            return history

        for run_dir in detect_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith('train'):
                try:
                    results_file = run_dir / "results.csv"
                    args_file = run_dir / "args.yaml"

                    if results_file.exists():
                        # Read training results
                        df = pd.read_csv(results_file)

                        # Read training arguments
                        args = {}
                        if args_file.exists():
                            with open(args_file, 'r') as f:
                                args = yaml.safe_load(f)

                        # Get model info
                        best_model = run_dir / "weights" / "best.pt"
                        last_model = run_dir / "weights" / "last.pt"

                        history.append({
                            "run_name": run_dir.name,
                            "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                            "epochs": len(df),
                            "best_model_path": str(best_model) if best_model.exists() else None,
                            "last_model_path": str(last_model) if last_model.exists() else None,
                            "final_metrics": {
                                "train_loss": float(df.iloc[-1].get('train/box_loss', 0)),
                                "val_loss": float(df.iloc[-1].get('val/box_loss', 0)),
                                "map50": float(df.iloc[-1].get('metrics/mAP50(B)', 0)),
                                "map50_95": float(df.iloc[-1].get('metrics/mAP50-95(B)', 0)),
                            },
                            "args": args,
                            "results_path": str(results_file)
                        })
                except Exception as e:
                    print(f"Error processing run {run_dir.name}: {e}")
                    continue

        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history

    def get_training_metrics(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific training run"""
        run_dir = self.runs_dir / "detect" / run_name
        results_file = run_dir / "results.csv"

        if not results_file.exists():
            return None

        try:
            df = pd.read_csv(results_file)

            # Extract key metrics
            metrics = {
                "epochs": list(range(1, len(df) + 1)),
                "train_box_loss": df.get('train/box_loss', []).tolist(),
                "train_cls_loss": df.get('train/cls_loss', []).tolist(),
                "train_dfl_loss": df.get('train/dfl_loss', []).tolist(),
                "val_box_loss": df.get('val/box_loss', []).tolist(),
                "val_cls_loss": df.get('val/cls_loss', []).tolist(),
                "val_dfl_loss": df.get('val/dfl_loss', []).tolist(),
                "map50": df.get('metrics/mAP50(B)', []).tolist(),
                "map50_95": df.get('metrics/mAP50-95(B)', []).tolist(),
                "precision": df.get('metrics/precision(B)', []).tolist(),
                "recall": df.get('metrics/recall(B)', []).tolist(),
                "lr0": df.get('lr/pg0', []).tolist(),
                "lr1": df.get('lr/pg1', []).tolist(),
                "lr2": df.get('lr/pg2', []).tolist(),
            }

            # Remove empty lists
            metrics = {k: v for k, v in metrics.items() if v}

            return metrics

        except Exception as e:
            print(f"Error reading metrics for {run_name}: {e}")
            return None

    def generate_training_plots(self, run_name: str) -> Dict[str, str]:
        """Generate interactive training plots using Plotly"""
        metrics = self.get_training_metrics(run_name)
        if not metrics:
            return {}

        plots = {}

        try:
            epochs = metrics.get('epochs', [])

            # Loss plots
            if any(k in metrics for k in ['train_box_loss', 'val_box_loss']):
                fig_loss = go.Figure()

                if 'train_box_loss' in metrics:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs,
                        y=metrics['train_box_loss'],
                        mode='lines+markers',
                        name='Train Box Loss',
                        line=dict(color='#FF6B6B')
                    ))

                if 'val_box_loss' in metrics:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs,
                        y=metrics['val_box_loss'],
                        mode='lines+markers',
                        name='Validation Box Loss',
                        line=dict(color='#4ECDC4')
                    ))

                fig_loss.update_layout(
                    title='Training and Validation Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_dark',
                    hovermode='x unified'
                )

                plots['loss'] = json.dumps(fig_loss, cls=PlotlyJSONEncoder)

            # mAP plots
            if any(k in metrics for k in ['map50', 'map50_95']):
                fig_map = go.Figure()

                if 'map50' in metrics:
                    fig_map.add_trace(go.Scatter(
                        x=epochs,
                        y=metrics['map50'],
                        mode='lines+markers',
                        name='mAP@0.5',
                        line=dict(color='#45B7D1')
                    ))

                if 'map50_95' in metrics:
                    fig_map.add_trace(go.Scatter(
                        x=epochs,
                        y=metrics['map50_95'],
                        mode='lines+markers',
                        name='mAP@0.5:0.95',
                        line=dict(color='#96CEB4')
                    ))

                fig_map.update_layout(
                    title='Mean Average Precision (mAP)',
                    xaxis_title='Epoch',
                    yaxis_title='mAP',
                    template='plotly_dark',
                    hovermode='x unified'
                )

                plots['map'] = json.dumps(fig_map, cls=PlotlyJSONEncoder)

            # Precision and Recall
            if any(k in metrics for k in ['precision', 'recall']):
                fig_pr = go.Figure()

                if 'precision' in metrics:
                    fig_pr.add_trace(go.Scatter(
                        x=epochs,
                        y=metrics['precision'],
                        mode='lines+markers',
                        name='Precision',
                        line=dict(color='#FECA57')
                    ))

                if 'recall' in metrics:
                    fig_pr.add_trace(go.Scatter(
                        x=epochs,
                        y=metrics['recall'],
                        mode='lines+markers',
                        name='Recall',
                        line=dict(color='#FF9FF3')
                    ))

                fig_pr.update_layout(
                    title='Precision and Recall',
                    xaxis_title='Epoch',
                    yaxis_title='Score',
                    template='plotly_dark',
                    hovermode='x unified'
                )

                plots['precision_recall'] = json.dumps(fig_pr, cls=PlotlyJSONEncoder)

            # Learning Rate
            if any(k in metrics for k in ['lr0', 'lr1', 'lr2']):
                fig_lr = go.Figure()

                for i, lr_key in enumerate(['lr0', 'lr1', 'lr2']):
                    if lr_key in metrics:
                        fig_lr.add_trace(go.Scatter(
                            x=epochs,
                            y=metrics[lr_key],
                            mode='lines+markers',
                            name=f'Learning Rate {i}',
                            line=dict(color=['#A8E6CF', '#FFD93D', '#DDA0DD'][i])
                        ))

                fig_lr.update_layout(
                    title='Learning Rate Schedule',
                    xaxis_title='Epoch',
                    yaxis_title='Learning Rate',
                    template='plotly_dark',
                    hovermode='x unified',
                    yaxis_type='log'
                )

                plots['learning_rate'] = json.dumps(fig_lr, cls=PlotlyJSONEncoder)

        except Exception as e:
            print(f"Error generating plots: {e}")

        return plots

    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare all trained models"""
        history = self.get_training_history()

        if not history:
            return {"models": [], "comparison_plot": None}

        # Prepare data for comparison
        comparison_data = []
        for run in history:
            metrics = run.get('final_metrics', {})
            comparison_data.append({
                'run_name': run['run_name'],
                'timestamp': run['timestamp'],
                'epochs': run['epochs'],
                'map50': metrics.get('map50', 0),
                'map50_95': metrics.get('map50_95', 0),
                'train_loss': metrics.get('train_loss', 0),
                'val_loss': metrics.get('val_loss', 0),
            })

        # Create comparison plot
        comparison_plot = None
        try:
            if comparison_data:
                df = pd.DataFrame(comparison_data)

                fig = go.Figure()

                # mAP50 comparison
                fig.add_trace(go.Bar(
                    x=df['run_name'],
                    y=df['map50'],
                    name='mAP@0.5',
                    marker_color='#4ECDC4'
                ))

                fig.update_layout(
                    title='Model Performance Comparison (mAP@0.5)',
                    xaxis_title='Training Run',
                    yaxis_title='mAP@0.5',
                    template='plotly_dark',
                    xaxis_tickangle=-45
                )

                comparison_plot = json.dumps(fig, cls=PlotlyJSONEncoder)

        except Exception as e:
            print(f"Error creating comparison plot: {e}")

        return {
            "models": comparison_data,
            "comparison_plot": comparison_plot
        }

    def export_training_data(self, format: str = "zip") -> str:
        """Export training data in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"training_data_export_{timestamp}"

        if format == "zip":
            import zipfile

            zip_path = self.results_dir / f"{export_name}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add training data
                for root, dirs, files in os.walk(self.base_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.base_dir)
                        zipf.write(file_path, arcname)

                # Add training runs
                if self.runs_dir.exists():
                    for root, dirs, files in os.walk(self.runs_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = Path("runs") / file_path.relative_to(self.runs_dir)
                            zipf.write(file_path, arcname)

            return str(zip_path)

        return ""

    def cleanup_old_runs(self, keep_latest: int = 5):
        """Clean up old training runs, keeping only the latest N"""
        history = self.get_training_history()

        if len(history) <= keep_latest:
            return

        runs_to_delete = history[keep_latest:]

        for run in runs_to_delete:
            run_dir = self.runs_dir / "detect" / run['run_name']
            if run_dir.exists():
                try:
                    shutil.rmtree(run_dir)
                    print(f"Deleted old training run: {run['run_name']}")
                except Exception as e:
                    print(f"Error deleting run {run['run_name']}: {e}")

    def get_dataset_analysis(self) -> Dict[str, Any]:
        """Analyze the training dataset"""
        images_dir = self.base_dir / "images"
        labels_dir = self.base_dir / "labels"
        classes_file = self.base_dir / "classes.txt"

        analysis = {
            "total_images": 0,
            "total_annotations": 0,
            "class_distribution": {},
            "image_sizes": [],
            "annotation_stats": {},
            "class_distribution_plot": None
        }

        try:
            # Count images
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                analysis["total_images"] = len(image_files)

                # Analyze image sizes
                from PIL import Image
                sizes = []
                for img_file in image_files[:100]:  # Sample first 100 images
                    try:
                        with Image.open(img_file) as img:
                            sizes.append({"width": img.width, "height": img.height})
                    except Exception:
                        continue

                analysis["image_sizes"] = sizes

            # Analyze annotations
            if labels_dir.exists() and classes_file.exists():
                # Load class names
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]

                class_counts = {cls: 0 for cls in classes}
                total_annotations = 0
                bbox_areas = []

                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if 0 <= class_id < len(classes):
                                        class_counts[classes[class_id]] += 1
                                        total_annotations += 1

                                        # Calculate bbox area (normalized)
                                        width, height = float(parts[3]), float(parts[4])
                                        bbox_areas.append(width * height)

                analysis["total_annotations"] = total_annotations
                analysis["class_distribution"] = class_counts
                analysis["annotation_stats"] = {
                    "avg_bbox_area": np.mean(bbox_areas) if bbox_areas else 0,
                    "min_bbox_area": np.min(bbox_areas) if bbox_areas else 0,
                    "max_bbox_area": np.max(bbox_areas) if bbox_areas else 0,
                }

                # Create class distribution plot
                if class_counts:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(class_counts.keys()),
                            y=list(class_counts.values()),
                            marker_color='#4ECDC4'
                        )
                    ])

                    fig.update_layout(
                        title='Class Distribution in Training Dataset',
                        xaxis_title='Class',
                        yaxis_title='Number of Annotations',
                        template='plotly_dark',
                        xaxis_tickangle=-45
                    )

                    analysis["class_distribution_plot"] = json.dumps(fig, cls=PlotlyJSONEncoder)

        except Exception as e:
            print(f"Error analyzing dataset: {e}")

        return analysis





