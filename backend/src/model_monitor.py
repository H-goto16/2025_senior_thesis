import time
import os
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from yolo.object_detection import YoloDetector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelFileHandler(FileSystemEventHandler):
    """モデルファイルの変更を監視するハンドラー"""

    def __init__(self, yolo_detector: YoloDetector, model_paths: list[str]):
        self.yolo_detector = yolo_detector
        self.model_paths = [Path(path) for path in model_paths]
        self.last_modified = {}

        # 初期のファイル時刻を記録
        for path in self.model_paths:
            if path.exists():
                self.last_modified[str(path)] = path.stat().st_mtime
                logger.info(f"Monitoring model file: {path}")

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # 監視対象のファイルかチェック
        if not any(file_path.samefile(model_path) for model_path in self.model_paths):
            return

        # ファイルの変更時刻をチェック
        try:
            current_mtime = file_path.stat().st_mtime
            last_mtime = self.last_modified.get(str(file_path), 0)

            # 変更が検出された場合
            if current_mtime > last_mtime:
                logger.info(f"Model file change detected: {file_path}")
                logger.info(f"Previous modification time: {last_mtime}")
                logger.info(f"Current modification time: {current_mtime}")

                # 少し待ってから再読み込み（ファイル書き込み完了を待つ）
                time.sleep(2)

                try:
                    # モデルを再読み込み
                    logger.info(f"Auto-reloading model from: {file_path}")
                    self.yolo_detector.reload_model(str(file_path))
                    logger.info(f"Model auto-reload completed successfully from: {file_path}")

                    # 更新時刻を記録
                    self.last_modified[str(file_path)] = current_mtime

                except Exception as e:
                    logger.error(f"Error during auto-reload: {e}")

        except Exception as e:
            logger.error(f"Error checking file modification: {e}")


class ModelMonitor:
    """モデルファイルの監視を行うクラス"""

    def __init__(self, yolo_detector: YoloDetector, model_paths: list[str] = None):
        self.yolo_detector = yolo_detector

        # デフォルトの監視パス
        if model_paths is None:
            self.model_paths = [
                "./yolov8s-world.pt"
            ]
        else:
            self.model_paths = model_paths

        self.observer = Observer()
        self.event_handler = ModelFileHandler(yolo_detector, self.model_paths)
        self.is_monitoring = False

        logger.info(f"ModelMonitor initialized with paths: {self.model_paths}")

    def start_monitoring(self):
        """監視を開始"""
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return

        try:
            # 各ディレクトリを監視
            for model_path in self.model_paths:
                path = Path(model_path)
                if path.exists():
                    directory = path.parent
                    logger.info(f"Starting to monitor directory: {directory}")
                    self.observer.schedule(self.event_handler, str(directory), recursive=False)
                else:
                    logger.warning(f"Model path does not exist: {model_path}")

            self.observer.start()
            self.is_monitoring = True
            logger.info("Model monitoring started successfully")

        except Exception as e:
            logger.error(f"Error starting model monitoring: {e}")
            raise

    def stop_monitoring(self):
        """監視を停止"""
        if not self.is_monitoring:
            logger.warning("Monitoring is not active")
            return

        try:
            self.observer.stop()
            self.observer.join()
            self.is_monitoring = False
            logger.info("Model monitoring stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping model monitoring: {e}")
            raise

    def get_status(self):
        """監視状態を取得"""
        return {
            "is_monitoring": self.is_monitoring,
            "monitored_paths": self.model_paths,
            "observer_alive": self.observer.is_alive() if hasattr(self.observer, 'is_alive') else False
        }


def start_model_monitoring(yolo_detector: YoloDetector, model_paths: list[str] = None):
    """モデル監視を開始する便利関数"""
    monitor = ModelMonitor(yolo_detector, model_paths)
    monitor.start_monitoring()
    return monitor


if __name__ == "__main__":
    # テスト用
    from yolo.object_detection import YoloDetector

    yolo = YoloDetector()
    monitor = start_model_monitoring(yolo)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("Monitoring stopped by user")
