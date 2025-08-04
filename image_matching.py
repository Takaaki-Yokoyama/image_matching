import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os

class ImageMatcher:
    """
    白黒画像の一部分から全体画像内での位置を特定するクラス
    """
    
    def __init__(self):
        self.template_methods = {
            'TM_CCOEFF': cv2.TM_CCOEFF,
            'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
            'TM_CCORR': cv2.TM_CCORR,
            'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
            'TM_SQDIFF': cv2.TM_SQDIFF,
            'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
        }
    
    def load_images(self, main_image_path: str, template_image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像を読み込み、グレースケールに変換
        
        Args:
            main_image_path: 全体画像のパス
            template_image_path: テンプレート画像（一部分）のパス
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (全体画像, テンプレート画像)
        """
        # 画像を読み込み
        main_img = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)
        template_img = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        
        if main_img is None:
            raise FileNotFoundError(f"全体画像が見つかりません: {main_image_path}")
        if template_img is None:
            raise FileNotFoundError(f"テンプレート画像が見つかりません: {template_image_path}")
            
        return main_img, template_img
    
    def template_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                         method: str = 'TM_CCOEFF_NORMED') -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        テンプレートマッチングを実行
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            method: マッチング手法
            
        Returns:
            Tuple[np.ndarray, Tuple[int, int], float]: (結果マップ, 最適位置, 最適値)
        """
        if method not in self.template_methods:
            raise ValueError(f"サポートされていないメソッド: {method}")
        
        # テンプレートマッチング実行
        result = cv2.matchTemplate(main_img, template_img, self.template_methods[method])
        
        # 最適位置を取得
        if method in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            best_match_loc = min_loc
            best_match_val = min_val
        else:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            best_match_loc = max_loc
            best_match_val = max_val
        
        return result, best_match_loc, best_match_val
    
    def find_multiple_matches(self, main_img: np.ndarray, template_img: np.ndarray, 
                            method: str = 'TM_CCOEFF_NORMED', threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        複数のマッチング位置を検出
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            method: マッチング手法
            threshold: 閾値
            
        Returns:
            List[Tuple[int, int, float]]: [(x, y, confidence), ...]
        """
        result, _, _ = self.template_matching(main_img, template_img, method)
        
        # 閾値以上の位置を検出
        if method in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
            locations = np.where(result <= 1 - threshold)
            confidences = 1 - result[locations]
        else:
            locations = np.where(result >= threshold)
            confidences = result[locations]
        
        matches = []
        for pt, conf in zip(zip(*locations[::-1]), confidences):
            matches.append((pt[0], pt[1], conf))
        
        return matches
    
    def visualize_results(self, main_img: np.ndarray, template_img: np.ndarray, 
                         matches: List[Tuple[int, int, float]], save_path: Optional[str] = None):
        """
        マッチング結果を可視化
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            matches: マッチング結果のリスト
            save_path: 保存パス（オプション）
        """
        h, w = template_img.shape
        
        # カラー画像に変換（可視化のため）
        result_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        
        # マッチング位置に矩形を描画
        for i, (x, y, confidence) in enumerate(matches):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # 最初のマッチは緑、その他は赤
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_img, f'{confidence:.3f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 結果を表示
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(main_img, cmap='gray')
        plt.title('全体画像')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(template_img, cmap='gray')
        plt.title('テンプレート画像')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('マッチング結果')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    メイン関数 - サンプル実行
    """
    matcher = ImageMatcher()
    
    # 画像パスを指定（実際のパスに変更してください）
    main_image_path = "main_image.png"  # 全体画像
    template_image_path = "template_image.png"  # 一部分画像
    
    try:
        # 画像を読み込み
        main_img, template_img = matcher.load_images(main_image_path, template_image_path)
        
        print(f"全体画像サイズ: {main_img.shape}")
        print(f"テンプレート画像サイズ: {template_img.shape}")
        
        # 単一の最適マッチを検出
        result, best_loc, best_val = matcher.template_matching(main_img, template_img)
        print(f"最適マッチ位置: {best_loc}, 信頼度: {best_val}")
        
        # 複数マッチを検出
        matches = matcher.find_multiple_matches(main_img, template_img, threshold=0.7)
        print(f"検出されたマッチ数: {len(matches)}")
        
        for i, (x, y, conf) in enumerate(matches):
            print(f"マッチ {i+1}: 位置({x}, {y}), 信頼度: {conf:.3f}")
        
        # 結果を可視化
        matcher.visualize_results(main_img, template_img, matches)
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("サンプル画像を作成します...")
        create_sample_images()

def create_sample_images():
    """
    テスト用のサンプル画像を作成
    """
    # サンプルの全体画像を作成
    main_img = np.random.randint(0, 256, (400, 600), dtype=np.uint8)
    
    # 特徴的なパターンを追加
    cv2.rectangle(main_img, (100, 100), (200, 150), 255, -1)
    cv2.rectangle(main_img, (150, 125), (175, 140), 0, -1)
    cv2.circle(main_img, (300, 200), 50, 128, -1)
    cv2.line(main_img, (400, 50), (500, 150), 200, 3)
    
    # テンプレート画像（全体画像の一部分）を作成
    template_img = main_img[110:140, 110:190]  # 矩形の一部を切り出し
    
    # 画像を保存
    cv2.imwrite("main_image.png", main_img)
    cv2.imwrite("template_image.png", template_img)
    
    print("サンプル画像を作成しました:")
    print("- main_image.png (全体画像)")
    print("- template_image.png (テンプレート画像)")

if __name__ == "__main__":
    main()
