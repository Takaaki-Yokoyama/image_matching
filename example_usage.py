"""
高精度画像マッチングの使用例
形状が少し違っても白点の分布状態が似ていれば検出可能
"""

from image_matching import ImageMatcher
import cv2
import numpy as np

def main():
    matcher = ImageMatcher()
    
    # 画像を読み込み
    main_img, template_img = matcher.load_images("main_image.png", "template_image.png")
    
    print("=== 高精度画像マッチング使用例 ===\n")
    
    # 1. 特徴分布マッチング（推奨：形状が違っても検出可能）
    print("1. 特徴分布マッチング（形状違いに強い）:")
    matches = matcher.find_multiple_matches(
        main_img, template_img,
        method='FEATURE_DISTRIBUTION',
        threshold=0.6,  # 低めの閾値で柔軟に検出
        use_advanced=True,
        step_size=2  # 高速化のため
    )
    print(f"   検出数: {len(matches)}")
    for i, (x, y, conf) in enumerate(matches[:3]):
        print(f"   位置{i+1}: ({x}, {y}), 信頼度: {conf:.3f}")
    
    # 2. 二値パターンマッチング（白点分布に特化）
    print("\n2. 二値パターンマッチング（白点分布重視）:")
    matches = matcher.find_multiple_matches(
        main_img, template_img,
        method='BINARY_PATTERN',
        threshold=0.5,  # 白点分布の類似度
        use_advanced=True,
        step_size=1
    )
    print(f"   検出数: {len(matches)}")
    for i, (x, y, conf) in enumerate(matches[:3]):
        print(f"   位置{i+1}: ({x}, {y}), 信頼度: {conf:.3f}")
    
    # 3. エッジ相関マッチング（エッジパターンを重視）
    print("\n3. エッジ相関マッチング（輪郭重視）:")
    matches = matcher.find_multiple_matches(
        main_img, template_img,
        method='EDGE_CORRELATION',
        threshold=0.4,
        use_advanced=True,
        step_size=2
    )
    print(f"   検出数: {len(matches)}")
    for i, (x, y, conf) in enumerate(matches[:3]):
        print(f"   位置{i+1}: ({x}, {y}), 信頼度: {conf:.3f}")
    
    # 4. カスタム閾値での検出例
    print("\n4. カスタム閾値での検出:")
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        matches = matcher.find_multiple_matches(
            main_img, template_img,
            method='FEATURE_DISTRIBUTION',
            threshold=thresh,
            use_advanced=True,
            step_size=3
        )
        print(f"   閾値{thresh}: {len(matches)}個検出")
    
    print("\n=== 使用方法のまとめ ===")
    print("- 形状が違っても検出したい → FEATURE_DISTRIBUTION (threshold=0.4-0.7)")
    print("- 白点の分布パターンを重視 → BINARY_PATTERN (threshold=0.3-0.6)")
    print("- 輪郭パターンを重視 → EDGE_CORRELATION (threshold=0.3-0.6)")
    print("- 厳密なマッチング → TM_CCOEFF_NORMED (threshold=0.7-0.9)")
    print("- 高速化が必要 → step_size=2-4に設定")

if __name__ == "__main__":
    main()
