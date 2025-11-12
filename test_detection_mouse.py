#!/usr/bin/env python3
"""
test_mouseディレクトリの画像に対して検出を実行するスクリプト
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any

API_BASE = "http://localhost:8000"

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

def main():
    """メイン処理"""
    print("=" * 80)
    print("🐭 Mouse Detection Test")
    print("=" * 80)

    # テストディレクトリ
    test_dir = Path("assets/mouse/test_mouse")

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return

    # 画像ファイルを取得
    image_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))

    if not image_files:
        print(f"❌ No images found in {test_dir}")
        return

    print(f"📸 Found {len(image_files)} images")
    print()

    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...")
        result = detect_image(image_path)
        result["image_name"] = image_path.name
        result["image_path"] = str(image_path)
        results.append(result)

        if "error" not in result:
            detections = result.get("detections", [])
            if detections:
                print(f"  ✅ Found {len(detections)} detection(s)")
                for det in detections:
                    print(f"     - {det['class']}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
            else:
                print(f"  ⚠️  No detections")
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
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
            successful_images += 1
            detections = result.get("detections", [])
            total_detections += len(detections)

            if detections:
                print(f"✅ {image_name}: {len(detections)} detection(s)")
                for det in detections:
                    print(f"   - {det['class']}: confidence={det['confidence']:.3f}")
            else:
                print(f"⚠️  {image_name}: No detections")

    print()
    print("=" * 80)
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful_images}")
    print(f"Failed: {failed_images}")
    print(f"Total detections: {total_detections}")
    print("=" * 80)

    # 結果をJSONファイルに保存
    output_file = Path("detection_results_test_mouse.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()

