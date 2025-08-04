"""
画像マッチングの実践例
このスクリプトは実際の画像ファイルを使用して画像マッチングを実行する例を示します。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from image_matching import ImageMatcher
from advanced_matching import AdvancedImageMatcher

def create_test_images():
    """
    テスト用の画像を作成する関数
    """
    print("テスト画像を作成中...")
    
    # より複雑な全体画像を作成
    main_img = np.zeros((500, 700), dtype=np.uint8)
    
    # 背景にノイズを追加
    noise = np.random.normal(128, 30, main_img.shape).astype(np.uint8)
    main_img = cv2.addWeighted(main_img, 0.3, noise, 0.7, 0)
    
    # 様々な図形を描画
    # 矩形
    cv2.rectangle(main_img, (50, 50), (150, 100), 255, -1)
    cv2.rectangle(main_img, (70, 70), (130, 90), 0, -1)
    
    # 円
    cv2.circle(main_img, (300, 150), 40, 200, -1)
    cv2.circle(main_img, (300, 150), 20, 100, -1)
    
    # 線
    cv2.line(main_img, (400, 100), (500, 200), 180, 5)
    cv2.line(main_img, (500, 100), (400, 200), 180, 5)
    
    # 多角形
    pts = np.array([[200, 300], [250, 280], [280, 320], [240, 350], [190, 340]], np.int32)
    cv2.fillPoly(main_img, [pts], 160)
    
    # テキスト
    cv2.putText(main_img, 'TEST', (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, 220, 3)
    
    # テンプレート画像を作成（全体画像の一部を切り出し）
    template1 = main_img[60:90, 60:140]  # 矩形の一部
    template2 = main_img[130:170, 280:320]  # 円の一部
    template3 = main_img[280:350, 440:550]  # テキストの一部
    
    # 画像を保存
    cv2.imwrite('test_main.png', main_img)
    cv2.imwrite('template_rect.png', template1)
    cv2.imwrite('template_circle.png', template2)
    cv2.imwrite('template_text.png', template3)
    
    print("作成された画像:")
    print("- test_main.png (全体画像)")
    print("- template_rect.png (矩形テンプレート)")
    print("- template_circle.png (円テンプレート)")
    print("- template_text.png (テキストテンプレート)")
    
    return main_img, template1, template2, template3

def basic_matching_demo():
    """
    基本的なマッチングのデモ
    """
    print("\n=== 基本的なマッチングのデモ ===")
    
    matcher = ImageMatcher()
    
    # テスト画像作成
    main_img, template1, template2, template3 = create_test_images()
    
    templates = [
        ('template_rect.png', '矩形テンプレート'),
        ('template_circle.png', '円テンプレート'), 
        ('template_text.png', 'テキストテンプレート')
    ]
    
    for template_path, description in templates:
        print(f"\n--- {description} ---")
        
        # 画像読み込み
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # 複数の手法でマッチング
        methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']
        
        for method in methods:
            result, best_loc, best_val = matcher.template_matching(main_img, template_img, method)
            print(f"  {method}: 位置{best_loc}, 信頼度{best_val:.3f}")
        
        # 複数マッチ検出
        matches = matcher.find_multiple_matches(main_img, template_img, threshold=0.7)
        print(f"  検出されたマッチ数: {len(matches)}")

def advanced_matching_demo():
    """
    高度なマッチング手法のデモ
    """
    print("\n=== 高度なマッチング手法のデモ ===")
    
    matcher = AdvancedImageMatcher()
    
    # 画像読み込み
    main_img = cv2.imread('test_main.png', cv2.IMREAD_GRAYSCALE)
    template_img = cv2.imread('template_rect.png', cv2.IMREAD_GRAYSCALE)
    
    if main_img is None or template_img is None:
        print("画像が見つかりません。create_test_images()を先に実行してください。")
        return
    
    # 1. マルチスケールマッチング
    print("\n--- マルチスケールマッチング ---")
    multi_matches = matcher.multi_scale_matching(main_img, template_img)
    for i, (x, y, conf, scale) in enumerate(multi_matches[:3]):
        print(f"  マッチ{i+1}: 位置({x}, {y}), 信頼度{conf:.3f}, スケール{scale:.2f}")
    
    # 2. 回転不変マッチング
    print("\n--- 回転不変マッチング ---")
    rotation_matches = matcher.rotation_invariant_matching(main_img, template_img, 
                                                          angles=range(0, 360, 30))
    for i, (x, y, conf, angle) in enumerate(rotation_matches[:3]):
        print(f"  マッチ{i+1}: 位置({x}, {y}), 信頼度{conf:.3f}, 角度{angle}°")
    
    # 3. エッジベースマッチング
    print("\n--- エッジベースマッチング ---")
    _, edge_loc, edge_conf = matcher.edge_based_matching(main_img, template_img)
    print(f"  位置: {edge_loc}, 信頼度: {edge_conf:.3f}")
    
    # 4. 手法比較
    print("\n--- 全手法比較 ---")
    results = matcher.compare_methods(main_img, template_img)
    for method, result in results.items():
        print(f"  {method}: 位置{result['location']}, 信頼度{result['confidence']:.3f}")

def interactive_demo():
    """
    インタラクティブなデモ
    """
    print("\n=== インタラクティブデモ ===")
    print("使用可能なコマンド:")
    print("1. basic - 基本的なマッチング")
    print("2. advanced - 高度なマッチング")
    print("3. create - テスト画像作成")
    print("4. visualize - 結果可視化")
    print("5. exit - 終了")
    
    matcher = ImageMatcher()
    
    while True:
        command = input("\nコマンドを入力してください: ").strip().lower()
        
        if command == 'exit':
            break
        elif command == 'create':
            create_test_images()
        elif command == 'basic':
            basic_matching_demo()
        elif command == 'advanced':
            advanced_matching_demo()
        elif command == 'visualize':
            try:
                # 画像読み込み
                main_img = cv2.imread('test_main.png', cv2.IMREAD_GRAYSCALE)
                template_img = cv2.imread('template_rect.png', cv2.IMREAD_GRAYSCALE)
                
                if main_img is None or template_img is None:
                    print("画像が見つかりません。'create'コマンドを先に実行してください。")
                    continue
                
                # マッチング実行
                matches = matcher.find_multiple_matches(main_img, template_img, threshold=0.6)
                
                # 結果可視化
                matcher.visualize_results(main_img, template_img, matches)
                
            except Exception as e:
                print(f"可視化エラー: {e}")
        else:
            print("無効なコマンドです。")

def benchmark_methods():
    """
    各手法の性能をベンチマーク
    """
    print("\n=== 手法性能ベンチマーク ===")
    
    import time
    
    # テスト画像準備
    main_img = cv2.imread('test_main.png', cv2.IMREAD_GRAYSCALE)
    template_img = cv2.imread('template_rect.png', cv2.IMREAD_GRAYSCALE)
    
    if main_img is None or template_img is None:
        print("画像が見つかりません。")
        return
    
    matcher = ImageMatcher()
    advanced_matcher = AdvancedImageMatcher()
    
    methods_to_test = [
        ("標準テンプレートマッチング", lambda: cv2.matchTemplate(main_img, template_img, cv2.TM_CCOEFF_NORMED)),
        ("エッジベースマッチング", lambda: advanced_matcher.edge_based_matching(main_img, template_img)),
        ("マルチスケールマッチング", lambda: advanced_matcher.multi_scale_matching(main_img, template_img)),
    ]
    
    for method_name, method_func in methods_to_test:
        # 実行時間測定
        start_time = time.time()
        try:
            result = method_func()
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"{method_name}: {execution_time:.4f}秒")
        except Exception as e:
            print(f"{method_name}: エラー - {e}")

def main():
    """
    メイン関数
    """
    print("画像マッチング実践例")
    print("=" * 50)
    
    # 必要なライブラリの確認
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ 必要なライブラリが利用可能です")
    except ImportError as e:
        print(f"✗ 必要なライブラリが不足しています: {e}")
        print("pip install opencv-python numpy matplotlib で追加してください")
        return
    
    # テスト画像が存在しない場合は作成
    if not os.path.exists('test_main.png'):
        print("テスト画像が見つかりません。作成します...")
        create_test_images()
    
    # デモ実行
    basic_matching_demo()
    advanced_matching_demo()
    benchmark_methods()
    
    print("\nインタラクティブモードを開始しますか？ (y/n): ", end="")
    if input().lower() == 'y':
        interactive_demo()

if __name__ == "__main__":
    main()
