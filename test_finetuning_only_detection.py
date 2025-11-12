#!/usr/bin/env python3
"""
ファインチューニングのみで検出できるか実験するスクリプト
- クラス追加なし
- ファインチューニング済みモデルをロード
- 検出を試す
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from yolo.object_detection import YoloDetector
from pathlib import Path

def test_finetuning_only_detection():
    """ファインチューニングのみで検出できるかテスト"""
    print("=" * 80)
    print("🔬 実験: ファインチューニングのみで検出できるか？")
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

    # 1. クラス追加なしでモデルを初期化
    print("Step 1: クラス追加なしでモデルを初期化")
    print("-" * 80)
    detector = YoloDetector()
    print(f"初期クラス: {detector.get_current_classes()}")
    print()

    # 2. ファインチューニング済みモデルをロード
    print("Step 2: ファインチューニング済みモデルをロード")
    print("-" * 80)
    detector.load_trained_model(str(best_model))
    print(f"ロード後のクラス: {detector.get_current_classes()}")
    print()

    # 3. モデルにクラス情報が含まれているか確認
    print("Step 3: モデルのクラス情報を確認")
    print("-" * 80)
    try:
        # YOLOモデルにはnames属性がある可能性がある
        if hasattr(detector.model, 'names'):
            print(f"モデルのnames属性: {detector.model.names}")
        if hasattr(detector.model, 'model') and hasattr(detector.model.model, 'names'):
            print(f"モデル内部のnames属性: {detector.model.model.names}")
    except Exception as e:
        print(f"クラス情報の確認中にエラー: {e}")
    print()

    # 4. クラス追加なしで検出を試す
    print("Step 4: クラス追加なしで検出を試す")
    print("-" * 80)
    test_image = Path("assets/mouse/test_mouse/mouse_3.jpg")

    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return

    print(f"テスト画像: {test_image}")
    print(f"現在のクラス: {detector.get_current_classes()}")
    print()

    try:
        result = detector.predict_image(str(test_image))

        if result is None:
            print("❌ 検出失敗: クラスが設定されていないため検出できません")
            print()
            print("結論: ファインチューニングのみでは検出できません")
            print("理由: YOLO-Worldは`set_classes()`でクラスを設定する必要があります")
        else:
            print("✅ 検出成功！")
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                print(f"検出数: {len(result.boxes)}")
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                    print(f"  [{i+1}] {cls_name}: confidence={conf:.3f}")
            else:
                print("検出結果: 検出なし")
    except Exception as e:
        print(f"❌ 検出中にエラー: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)

if __name__ == "__main__":
    test_finetuning_only_detection()

