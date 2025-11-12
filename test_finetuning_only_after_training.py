#!/usr/bin/env python3
"""
学習完了後、クラス追加なしでファインチューニングのみで検出テスト
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from yolo.object_detection import YoloDetector
from pathlib import Path
import json

def test_finetuning_only_detection():
    """ファインチューニングのみで検出できるかテスト"""
    print("=" * 80)
    print("🔬 実験: ファインチューニングのみで検出できるか？（50枚、100エポック）")
    print("=" * 80)
    print()

    # 最新の学習済みモデルを探す
    runs_dir = Path("/home/haruki-goto/workspace/lab/dish_detection/runs/detect")
    if not runs_dir.exists():
        print("❌ runs/detect directory not found")
        return

    train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')],
                       key=lambda x: x.stat().st_mtime, reverse=True)

    if not train_dirs:
        print("❌ No training runs found")
        return

    latest_run = train_dirs[0]
    best_model = latest_run / "weights" / "best.pt"

    if not best_model.exists():
        print(f"❌ Best model not found: {best_model}")
        return

    print(f"📦 Using trained model: {best_model}")
    print()

    # custom_vocab.jsonを一時的にリネーム
    vocab_file = Path("backend/src/custom_vocab.json")
    vocab_backup = None

    if vocab_file.exists():
        vocab_backup = vocab_file.with_suffix('.json.backup')
        shutil.move(str(vocab_file), str(vocab_backup))
        print(f"📝 custom_vocab.jsonを一時的にリネーム: {vocab_backup}")
        print()

    try:
        # クラス追加なしでモデルを初期化
        print("Step 1: クラス追加なしでモデルを初期化")
        print("-" * 80)
        detector = YoloDetector(vocab_file="nonexistent_vocab.json")
        print(f"初期クラス: {detector.get_current_classes()}")
        print()

        # ファインチューニング済みモデルをロード
        print("Step 2: ファインチューニング済みモデルをロード")
        print("-" * 80)
        detector.load_trained_model(str(best_model))
        print(f"ロード後のクラス: {detector.get_current_classes()}")
        print()

        # モデルのクラス情報を確認
        print("Step 3: モデルのクラス情報を確認")
        print("-" * 80)
        model_names = None
        try:
            if hasattr(detector.model, 'names'):
                model_names = detector.model.names
                print(f"モデルのnames属性: {model_names}")
        except Exception as e:
            print(f"クラス情報の確認中にエラー: {e}")
        print()

        # test_mouseの10枚で検出テスト
        print("Step 4: test_mouseの10枚で検出テスト（クラス追加なし）")
        print("-" * 80)
        test_dir = Path("assets/mouse/test_mouse")

        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            return

        image_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))

        if not image_files:
            print(f"❌ No images found in {test_dir}")
            return

        print(f"テスト画像: {len(image_files)}枚")
        print()

        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing {image_path.name}...")

            # 直接モデルで検出（predict_imageのチェックを回避）
            try:
                if model_names:
                    results_direct = detector.model.predict(str(image_path), conf=0.25, device=detector.device, verbose=False)
                    result = results_direct[0]

                    detections = []
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                            bbox = [float(coord) for coord in box.xyxy[0].tolist()]

                            detections.append({
                                "class": cls_name,
                                "confidence": conf,
                                "bbox": bbox
                            })

                        print(f"  ✅ Found {len(detections)} detection(s)")
                        for det in detections:
                            print(f"     - {det['class']}: confidence={det['confidence']:.3f}")
                    else:
                        print(f"  ⚠️  No detections")

                    results.append({
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "detections": detections
                    })
                else:
                    print(f"  ❌ Model names not available")
                    results.append({
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "detections": [],
                        "error": "Model names not available"
                    })
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "detections": [],
                    "error": str(e)
                })
            print()

        # サマリー
        print("=" * 80)
        print("📊 DETECTION SUMMARY")
        print("=" * 80)

        total_detections = 0
        successful_images = 0
        failed_images = 0

        for result in results:
            image_name = result.get("image_name", "unknown")
            if "error" in result:
                failed_images += 1
                print(f"❌ {image_name}: ERROR - {result.get('error', 'Unknown error')}")
            else:
                detections = result.get("detections", [])
                if detections:
                    successful_images += 1
                    total_detections += len(detections)
                    print(f"✅ {image_name}: {len(detections)} detection(s)")
                    for det in detections:
                        print(f"   - {det['class']}: confidence={det['confidence']:.3f}")
                else:
                    failed_images += 1
                    print(f"⚠️  {image_name}: No detections")

        print()
        print("=" * 80)
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful_images}")
        print(f"Failed: {failed_images}")
        print(f"Total detections: {total_detections}")
        print(f"Detection rate: {successful_images / len(results) * 100:.1f}%")
        print("=" * 80)

        # 結果をJSONファイルに保存
        output_file = Path("detection_results_finetuning_only_50images_100epochs.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

    finally:
        # custom_vocab.jsonを復元
        if vocab_backup and vocab_backup.exists():
            shutil.move(str(vocab_backup), str(vocab_file))
            print(f"📝 custom_vocab.jsonを復元しました")

    print()
    print("=" * 80)

if __name__ == "__main__":
    test_finetuning_only_detection()

