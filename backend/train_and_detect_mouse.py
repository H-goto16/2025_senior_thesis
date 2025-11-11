#!/usr/bin/env python3
"""
Mouse画像の学習と検出を実行するスクリプト
- Epoch50で学習を実行
- test_mouseの10枚の画像に対して検出を実行
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

API_BASE = "http://localhost:8000"
TIMEOUT = 3600  # 1時間のタイムアウト（学習用）

def check_server():
    """サーバーが起動しているか確認"""
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_training(epochs: int = 50) -> Dict[str, Any]:
    """学習を開始（非同期エンドポイントを使用）"""
    print(f"🚀 Starting training with {epochs} epochs...")

    try:
        # 非同期エンドポイントを使用してプロセスが落ちる問題を回避
        response = requests.post(
            f"{API_BASE}/training/start-async",
            params={"epochs": epochs},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Training started: {result['message']}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to start training: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        raise

def wait_for_training_completion(max_wait_time: int = 7200) -> bool:
    """学習の完了を待つ"""
    print("⏳ Waiting for training to complete...")
    start_time = time.time()
    training_started = False

    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{API_BASE}/training/status", timeout=5)
            response.raise_for_status()
            status = response.json()

            is_training = status.get("is_training", False)
            current_epoch = status.get("current_epoch", 0)
            total_epochs = status.get("total_epochs", 0)
            progress = status.get("progress", 0.0)
            status_message = status.get("status_message", "")

            # 学習が開始されたことを確認
            if is_training:
                training_started = True

            # 学習が完了したか確認
            # 1. is_trainingがFalseで、かつ学習が開始されていた
            # 2. status_messageが完了を示している
            # 3. current_epoch == total_epochs かつ is_training == False
            if training_started and not is_training:
                if total_epochs > 0 and current_epoch >= total_epochs:
                    print(f"✅ Training completed: Epoch {current_epoch}/{total_epochs} - {status_message}")
                    return True
                elif "completed" in status_message.lower() or "successfully" in status_message.lower():
                    print(f"✅ Training completed: {status_message}")
                    return True
                elif status_message == "Ready":
                    # "Ready"の場合は、学習が完了したか確認するため、もう少し待つ
                    print(f"⚠️ Status is 'Ready', checking if training actually completed...")
                    time.sleep(5)
                    # もう一度確認
                    response = requests.get(f"{API_BASE}/training/status", timeout=5)
                    response.raise_for_status()
                    status = response.json()
                    if not status.get("is_training", False):
                        print(f"✅ Training completed: {status_message}")
                        return True

            # 学習中の進捗を表示
            if is_training:
                if total_epochs > 0:
                    print(f"📊 Progress: Epoch {current_epoch}/{total_epochs} ({progress:.1f}%) - {status_message}")
                else:
                    print(f"📊 Status: {status_message}")
            elif not training_started:
                # まだ学習が開始されていない
                print(f"⏳ Waiting for training to start... (Status: {status_message})")

            time.sleep(10)  # 10秒ごとにチェック

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error checking training status: {e}")
            time.sleep(5)

    print("❌ Training timeout - exceeded max wait time")
    return False

def detect_image(image_path: Path) -> Dict[str, Any]:
    """画像に対して検出を実行"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_BASE}/detect",
                files=files,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to detect {image_path.name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return {"error": str(e), "detections": []}

def detect_test_images(test_dir: Path) -> List[Dict[str, Any]]:
    """test_mouseディレクトリ内の画像に対して検出を実行"""
    print(f"\n🔍 Starting detection on images in {test_dir}...")

    # 画像ファイルを取得
    image_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))

    if not image_files:
        print(f"❌ No images found in {test_dir}")
        return []

    print(f"📸 Found {len(image_files)} images")

    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
        result = detect_image(image_path)
        result["image_name"] = image_path.name
        result["image_path"] = str(image_path)
        results.append(result)

        if "error" not in result:
            detections = result.get("detections", [])
            print(f"  ✅ Found {len(detections)} detections")
            for det in detections:
                print(f"    - {det['class']}: {det['confidence']:.3f} at {det['bbox']}")
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown error')}")

        time.sleep(0.5)  # APIに負荷をかけないように少し待機

    return results

def print_summary(results: List[Dict[str, Any]]):
    """検出結果のサマリーを表示"""
    print("\n" + "=" * 80)
    print("📊 DETECTION SUMMARY")
    print("=" * 80)

    total_detections = 0
    successful_images = 0
    failed_images = 0

    for result in results:
        image_name = result.get("image_name", "unknown")
        if "error" in result:
            failed_images += 1
            print(f"\n❌ {image_name}: ERROR - {result.get('error', 'Unknown error')}")
        else:
            successful_images += 1
            detections = result.get("detections", [])
            total_detections += len(detections)

            if detections:
                print(f"\n✅ {image_name}: {len(detections)} detection(s)")
                for det in detections:
                    print(f"   - {det['class']}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
            else:
                print(f"\n⚠️ {image_name}: No detections")

    print("\n" + "=" * 80)
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {successful_images}")
    print(f"Failed: {failed_images}")
    print(f"Total detections: {total_detections}")
    print("=" * 80)

def main():
    """メイン処理"""
    print("=" * 80)
    print("🐭 Mouse Training and Detection Script")
    print("=" * 80)

    # サーバーの確認
    if not check_server():
        print("❌ Server is not running!")
        print(f"Please start the server: cd backend/src && uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print("✅ Server is running")

    # 学習データの確認
    try:
        response = requests.get(f"{API_BASE}/training/data/stats", timeout=5)
        response.raise_for_status()
        stats = response.json()
        print(f"\n📊 Training Data Stats:")
        print(f"  - Images: {stats.get('total_images', 0)}")
        print(f"  - Labels: {stats.get('total_labels', 0)}")
        print(f"  - Classes: {stats.get('classes', [])}")
    except Exception as e:
        print(f"⚠️ Could not get training data stats: {e}")

    # 学習を開始
    try:
        start_training(epochs=50)

        # 学習の完了を待つ（必須）
        print("\n" + "=" * 80)
        print("⏳ WAITING FOR TRAINING TO COMPLETE")
        print("=" * 80)
        training_completed = wait_for_training_completion(max_wait_time=10800)  # 3時間まで待機

        if not training_completed:
            print("\n❌ ERROR: Training did not complete within the timeout period.")
            print("❌ Cannot proceed with detection without completed training.")
            print("❌ Please check the training status manually and try again.")
            sys.exit(1)

        # 学習完了後、モデルがロードされるのを待つ
        print("\n⏳ Waiting for trained model to be loaded...")
        time.sleep(10)  # モデルのロードに時間がかかる可能性があるため、少し長めに待機

        # モデルがロードされたことを確認
        try:
            response = requests.get(f"{API_BASE}/model/info", timeout=5)
            response.raise_for_status()
            model_info = response.json()
            print(f"✅ Model loaded: {model_info.get('model_path', 'unknown')}")
            print(f"   Classes: {model_info.get('current_classes', [])}")
        except Exception as e:
            print(f"⚠️ Could not verify model info: {e}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("❌ Cannot proceed with detection without completed training.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # test_mouseディレクトリの画像に対して検出を実行
    # スクリプトの場所から相対パスで探す
    script_dir = Path(__file__).parent
    test_dir = script_dir.parent.parent / "assets" / "mouse" / "test_mouse"

    if not test_dir.exists():
        # 絶対パスでも試す
        test_dir = Path("/home/haruki-goto/workspace/lab/dish-detection/assets/mouse/test_mouse")

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Script directory: {script_dir}")
        sys.exit(1)

    results = detect_test_images(test_dir)

    # 結果を表示
    print_summary(results)

    # 結果をJSONファイルに保存
    output_file = Path("detection_results_mouse.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

