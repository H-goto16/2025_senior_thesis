from ultralytics import YOLOWorld, YOLO
import json
from pathlib import Path
import os
import shutil
from datetime import datetime
import yaml
import torch

class YoloDetector:
    def __init__(self, model_path="./yolov8s-world.pt", vocab_file="custom_vocab.json"):
        # Resolve device first before loading model
        self.device = self._resolve_device()
        # Load model with explicit device to avoid CUDA issues
        print(f"Initializing YOLO model on device: {self.device}")
        self.model = YOLOWorld(model_path)
        self.model_path = model_path
        self.vocab_file = Path(vocab_file)
        self.current_classes = set()
        self._load_custom_vocab()

    def _resolve_device(self) -> str:
        """Select a safe device. Prefer CPU unless CUDA is both available and usable.

        Older GPUs (e.g., sm_52) are often incompatible with the installed PyTorch builds.
        To avoid runtime 500s, default to CPU and only opt into CUDA when explicitly
        requested via env and capability is sufficient.
        """
        requested = os.getenv("YOLO_DEVICE") or os.getenv("YW_DEVICE")
        if requested:
            if requested.startswith("cuda"):
                if self._is_cuda_usable():
                    return requested
                print("CUDA requested but not usable; falling back to CPU.")
                return "cpu"
            return requested

        return "cuda:0" if self._is_cuda_usable() else "cpu"

    def _is_cuda_usable(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(0)
            capability = major * 10 + minor

            # Check PyTorch CUDA version
            cuda_version = torch.version.cuda
            if cuda_version:
                # CUDA 12.x requires Compute Capability 7.0+ (Ampere or newer)
                # CUDA 11.x supports Compute Capability 3.5+ (Kepler or newer)
                if cuda_version.startswith("12."):
                    # CUDA 12.x: Only support Compute Capability 7.0+ (RTX 30 series, A100, etc.)
                    if capability < 70:
                        print(f"⚠️ GPU Compute Capability {capability/10:.1f} is not supported by CUDA {cuda_version}")
                        print(f"⚠️ CUDA 12.x requires Compute Capability 7.0+ (Ampere or newer)")
                        print(f"⚠️ Falling back to CPU")
                        return False
                elif cuda_version.startswith("11."):
                    # CUDA 11.x: Support Compute Capability 3.5+ (GTX980Ti is 5.2, so OK)
                    return capability >= 35

            # Default: Support Compute Capability 5.2+ for older CUDA versions
            return capability >= 52
        except Exception as e:
            print(f"⚠️ Error checking CUDA capability: {e}")
            return False

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
        try:
            results = self.model.predict(image_path, conf=conf_threshold, device=self.device, verbose=False)
            return results[0]
        except (RuntimeError, Exception) as e:
            error_msg = str(e).lower()
            # Check if it's a CUDA compatibility error
            is_cuda_error = (
                "cuda" in error_msg and
                ("kernel" in error_msg or "device" in error_msg or "capability" in error_msg or "no kernel image" in error_msg)
            )

            # Retry on CPU if a CUDA/device error occurs
            if self.device != "cpu" and (is_cuda_error or "cuda" in error_msg):
                print(f"⚠️ Device '{self.device}' failed with error: {e}")
                print("⚠️ Falling back to CPU for detection...")
                try:
                    # Force model to CPU by moving it explicitly
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                        self.model.model.to('cpu')
                    # Update device setting
                    self.device = "cpu"
                    # Retry prediction on CPU
                    results = self.model.predict(image_path, conf=conf_threshold, device="cpu", verbose=False)
                    print("✅ Detection completed on CPU")
                    return results[0]
                except Exception as cpu_error:
                    print(f"❌ CPU fallback also failed: {cpu_error}")
                    # Try to reload model on CPU as last resort
                    try:
                        print("⚠️ Attempting to reload model on CPU...")
                        model_path = self.model_path
                        self.model = YOLOWorld(model_path)
                        if self.current_classes:
                            self._update_model_classes()
                        self.device = "cpu"
                        results = self.model.predict(image_path, conf=conf_threshold, device="cpu", verbose=False)
                        print("✅ Detection completed after model reload on CPU")
                        return results[0]
                    except Exception as reload_error:
                        print(f"❌ Model reload also failed: {reload_error}")
                        raise cpu_error
            elif self.device == "cpu":
                # Already on CPU, just re-raise
                raise
            else:
                # Other error, re-raise
                raise

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
            print(f"Using device: {self.device}")

            # Calculate optimal batch size based on data size
            data_dir = Path(data_config_path).parent
            images_dir = data_dir / "images"
            image_count = 0
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
                # Use smaller batch size to reduce memory usage, especially on CPU
                if self.device.startswith("cuda"):
                    batch_size = min(16, max(8, image_count // 4))
                else:
                    # Reduce batch size for CPU to prevent memory issues
                    batch_size = min(4, max(2, image_count // 8))
                print(f"Detected {image_count} images, using batch_size={batch_size}")
            else:
                batch_size = 8 if self.device.startswith("cuda") else 2
                print(f"Using default batch_size={batch_size}")

            # Set workers to 0 to avoid multiprocessing issues and reduce memory usage
            # YOLO may automatically set workers=0 in certain environments anyway
            workers = 0
            print(f"Using workers={workers} for data loading (single-threaded to reduce memory usage)")

            # Start training with error handling for CUDA compatibility
            try:
                # Optimize for memory-constrained environments
                train_kwargs = {
                    "data": data_config_path,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch_size,
                    "workers": workers,
                    "patience": 10,
                    "save": True,
                    "plots": False,  # Disable plots to save memory
                    "device": self.device,
                    "verbose": True,
                    "val": True,  # Explicitly enable validation
                    "save_period": 10,  # Save checkpoint every 10 epochs to prevent data loss
                    "cache": False,  # Disable caching to reduce memory usage
                    "amp": False,  # Disable mixed precision on CPU (not needed)
                }

                results = self.model.train(**train_kwargs)
            except RuntimeError as e:
                error_msg = str(e).lower()
                # Check if it's a CUDA compatibility error
                if "cuda" in error_msg and ("kernel" in error_msg or "device" in error_msg or "capability" in error_msg):
                    if self.device.startswith("cuda"):
                        print(f"⚠️ CUDA error detected: {e}")
                        print("⚠️ Falling back to CPU training (this will be slower but should work)")
                        # Recalculate batch size for CPU fallback (reduce to save memory)
                        if image_count > 0:
                            batch_size = min(4, max(2, image_count // 8))
                        else:
                            batch_size = 2
                        # Set workers to 0 for CPU fallback to reduce memory usage
                        workers = 0

                        # Retry on CPU with memory-optimized settings
                        train_kwargs = {
                            "data": data_config_path,
                            "epochs": epochs,
                            "imgsz": imgsz,
                            "batch": batch_size,
                            "workers": workers,
                            "patience": 10,
                            "save": True,
                            "plots": False,  # Disable plots to save memory
                            "device": "cpu",
                            "verbose": True,
                            "val": True,
                            "save_period": 10,  # Save checkpoint every 10 epochs to prevent data loss
                            "cache": False,  # Disable caching to reduce memory usage
                            "amp": False,  # Disable mixed precision on CPU
                        }
                        results = self.model.train(**train_kwargs)
                        # Update device to CPU for future operations
                        self.device = "cpu"
                    else:
                        raise
                else:
                    raise

            print("Fine-tuning completed successfully!")

            # Print training summary
            if results:
                try:
                    print("\n" + "=" * 80)
                    print("📊 TRAINING SUMMARY")
                    print("=" * 80)
                    if hasattr(results, 'results_dict'):
                        metrics = results.results_dict
                        print(f"Final mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}" if isinstance(metrics.get('metrics/mAP50(B)', None), (int, float)) else f"Final mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                        print(f"Final mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}" if isinstance(metrics.get('metrics/mAP50-95(B)', None), (int, float)) else f"Final mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
                        print(f"Final Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}" if isinstance(metrics.get('metrics/precision(B)', None), (int, float)) else f"Final Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
                        print(f"Final Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}" if isinstance(metrics.get('metrics/recall(B)', None), (int, float)) else f"Final Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
                    print("=" * 80 + "\n")
                except Exception as e:
                    print(f"⚠️ Could not print training summary: {e}")

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
            # Ensure device is set to CPU if CUDA is not usable
            if self.device.startswith("cuda") and not self._is_cuda_usable():
                print(f"⚠️ CUDA not usable, switching to CPU for model loading")
                self.device = "cpu"

            print(f"Loading model on device: {self.device}")
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

