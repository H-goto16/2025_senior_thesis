from ultralytics import YOLOWorld, YOLO
import json
from pathlib import Path
import os
import shutil
from datetime import datetime
import yaml

class YoloDetector:
    def __init__(self, model_path="./yolov8s-world.pt", vocab_file="custom_vocab.json"):
        self.model = YOLOWorld(model_path)
        self.model_path = model_path
        self.vocab_file = Path(vocab_file)
        self.current_classes = set()
        self._load_custom_vocab()

    def _load_custom_vocab(self):
        if self.vocab_file.exists():
            try:
                with open(self.vocab_file, 'r', encoding='utf-8') as f:
                    loaded_vocab = json.load(f)
                    if isinstance(loaded_vocab, list):
                        self.current_classes.update(loaded_vocab)
                        print(f"Loaded custom vocabulary: {self.current_classes}")
                    else:
                        print(f"Warning: Invalid format in {self.vocab_file}.")
            except json.JSONDecodeError:
                print(f"Warning: JSON decoding error in {self.vocab_file}. File might be corrupted.")
        self._update_model_classes()

    def _save_custom_vocab(self):
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.current_classes), f, ensure_ascii=False, indent=4)
        print(f"Custom vocabulary saved to {self.vocab_file}.")

    def _update_model_classes(self):
        if self.current_classes:
            self.model.set_classes(list(self.current_classes))
            print(f"Model detection classes updated: {list(self.current_classes)}")
        else:
            print("No detection classes set.")

    def add_classes(self, new_classes: list[str]):
        for cls in new_classes:
            self.current_classes.add(cls)
        self._update_model_classes()
        self._save_custom_vocab()

    def get_current_classes(self) -> list[str]:
        return list(self.current_classes)

    def predict_image(self, image_path: str, conf_threshold: float = 0.25):
        if not self.current_classes:
            print("Warning: No detection classes set. Please add classes using add_classes() first.")
            return None

        print(f"Executing detection on {image_path} (Classes: {list(self.current_classes)})...")
        results = self.model.predict(image_path, conf=conf_threshold, verbose=False)
        return results[0]

    def fine_tune_model(self, data_config_path: str, epochs: int = 50, imgsz: int = 640):
        """
        Fine-tune the YOLO model with custom labeled data

        Args:
            data_config_path: Path to the data.yaml configuration file
            epochs: Number of training epochs
            imgsz: Image size for training

        Returns:
            Training results
        """
        try:
            print(f"Starting fine-tuning with config: {data_config_path}")
            print(f"Training parameters: epochs={epochs}, imgsz={imgsz}")

            # Start training
            results = self.model.train(
                data=data_config_path,
                epochs=epochs,
                imgsz=imgsz,
                patience=10,
                save=True,
                plots=True,
                device='cpu',  # Use CPU for compatibility, change to 'cuda' if GPU available
                verbose=True
            )

            print("Fine-tuning completed successfully!")
            return results

        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            raise e

    def load_trained_model(self, model_path: str):
        """
        Load a fine-tuned model

        Args:
            model_path: Path to the trained model file
        """
        try:
            print(f"Loading fine-tuned model from: {model_path}")
            self.model = YOLOWorld(model_path)
            self.model_path = model_path

            # Update classes if they exist
            if self.current_classes:
                self._update_model_classes()

            print("Fine-tuned model loaded successfully!")

        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            raise e

    def get_model_info(self):
        """
        Get information about the current model
        """
        return {
            "model_path": self.model_path,
            "current_classes": list(self.current_classes),
            "model_type": type(self.model).__name__
        }

    def backup_current_model(self) -> str:
        """
        Create a backup of the current model

        Returns:
            Path to the backup file
        """
        try:
            backup_dir = Path("model_backups")
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"model_backup_{timestamp}.pt"
            backup_path = backup_dir / backup_name

            shutil.copy2(self.model_path, backup_path)

            # Also backup the vocabulary
            vocab_backup = backup_dir / f"vocab_backup_{timestamp}.json"
            if self.vocab_file.exists():
                shutil.copy2(self.vocab_file, vocab_backup)

            print(f"Model backed up to: {backup_path}")
            return str(backup_path)

        except Exception as e:
            print(f"Error backing up model: {e}")
            raise e

    def validate_model_performance(self, test_data_path: str = None) -> dict:
        """
        Validate model performance on test data

        Args:
            test_data_path: Path to test dataset configuration

        Returns:
            Validation metrics
        """
        try:
            if test_data_path and Path(test_data_path).exists():
                # Run validation on test dataset
                results = self.model.val(data=test_data_path, verbose=False)

                return {
                    "map50": float(results.box.map50),
                    "map50_95": float(results.box.map),
                    "precision": float(results.box.mp),
                    "recall": float(results.box.mr),
                    "test_data_path": test_data_path
                }
            else:
                # Basic model info if no test data
                return {
                    "model_loaded": True,
                    "classes_count": len(self.current_classes),
                    "model_path": self.model_path
                }

        except Exception as e:
            print(f"Error validating model: {e}")
            return {"error": str(e)}

    def get_available_models(self) -> list:
        """
        Get list of available trained models

        Returns:
            List of model information
        """
        models = []

        # Check runs directory for trained models
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith('train'):
                    best_model = run_dir / "weights" / "best.pt"
                    last_model = run_dir / "weights" / "last.pt"

                    if best_model.exists() or last_model.exists():
                        # Read training args if available
                        args_file = run_dir / "args.yaml"
                        args = {}
                        if args_file.exists():
                            try:
                                with open(args_file, 'r') as f:
                                    args = yaml.safe_load(f)
                            except:
                                pass

                        models.append({
                            "run_name": run_dir.name,
                            "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                            "best_model": str(best_model) if best_model.exists() else None,
                            "last_model": str(last_model) if last_model.exists() else None,
                            "args": args
                        })

        # Check backup directory
        backup_dir = Path("model_backups")
        if backup_dir.exists():
            for backup_file in backup_dir.glob("*.pt"):
                models.append({
                    "run_name": f"backup_{backup_file.stem}",
                    "timestamp": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    "best_model": str(backup_file),
                    "last_model": None,
                    "args": {"type": "backup"}
                })

        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models

