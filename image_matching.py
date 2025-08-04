import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os
from sklearn.metrics import normalized_mutual_info_score
from scipy.ndimage import gaussian_filter

class ImageMatcher:
    """
    白黒画像の一部分から全体画像内での位置を特定するクラス
    形状の違いに対応できる高精度マッチング機能付き
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
        
        self.advanced_methods = {
            'HISTOGRAM_CORRELATION': self._histogram_correlation,
            'EDGE_CORRELATION': self._edge_correlation,
            'GRADIENT_CORRELATION': self._gradient_correlation,
            'BINARY_PATTERN': self._binary_pattern_matching,
            'FEATURE_DISTRIBUTION': self._feature_distribution_matching
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
    
    def _histogram_correlation(self, main_img: np.ndarray, template_img: np.ndarray, 
                              region: Tuple[int, int, int, int]) -> float:
        """
        ヒストグラム相関によるマッチング評価
        """
        x, y, w, h = region
        if x + w > main_img.shape[1] or y + h > main_img.shape[0]:
            return 0.0
        
        roi = main_img[y:y+h, x:x+w]
        
        # ヒストグラムを計算
        hist1 = cv2.calcHist([template_img], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([roi], [0], None, [256], [0, 256])
        
        # 相関係数を計算
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)
    
    def _edge_correlation(self, main_img: np.ndarray, template_img: np.ndarray, 
                         region: Tuple[int, int, int, int]) -> float:
        """
        エッジ特徴による相関マッチング
        """
        x, y, w, h = region
        if x + w > main_img.shape[1] or y + h > main_img.shape[0]:
            return 0.0
        
        roi = main_img[y:y+h, x:x+w]
        
        # Sobelエッジ検出
        sobel_template = cv2.Sobel(template_img, cv2.CV_32F, 1, 1, ksize=3)
        sobel_roi = cv2.Sobel(roi, cv2.CV_32F, 1, 1, ksize=3)
        
        # 正規化（手動）
        sobel_template = np.abs(sobel_template)
        sobel_roi = np.abs(sobel_roi)
        
        if sobel_template.max() > 0:
            sobel_template = (sobel_template / sobel_template.max() * 255).astype(np.uint8)
        else:
            sobel_template = sobel_template.astype(np.uint8)
            
        if sobel_roi.max() > 0:
            sobel_roi = (sobel_roi / sobel_roi.max() * 255).astype(np.uint8)
        else:
            sobel_roi = sobel_roi.astype(np.uint8)
        
        # サイズチェック
        if sobel_roi.shape[0] < sobel_template.shape[0] or sobel_roi.shape[1] < sobel_template.shape[1]:
            return 0.0
        
        # 正規化相互相関
        try:
            correlation = cv2.matchTemplate(sobel_roi, sobel_template, cv2.TM_CCOEFF_NORMED)
            return max(0.0, correlation[0, 0] if correlation.size > 0 else 0.0)
        except:
            return 0.0
    
    def _gradient_correlation(self, main_img: np.ndarray, template_img: np.ndarray, 
                            region: Tuple[int, int, int, int]) -> float:
        """
        勾配方向による相関マッチング
        """
        x, y, w, h = region
        if x + w > main_img.shape[1] or y + h > main_img.shape[0]:
            return 0.0
        
        roi = main_img[y:y+h, x:x+w]
        
        # 勾配計算
        grad_x_t, grad_y_t = np.gradient(template_img.astype(float))
        grad_x_r, grad_y_r = np.gradient(roi.astype(float))
        
        # 勾配方向の類似度
        magnitude_t = np.sqrt(grad_x_t**2 + grad_y_t**2)
        magnitude_r = np.sqrt(grad_x_r**2 + grad_y_r**2)
        
        # 正規化
        magnitude_t = magnitude_t / (np.max(magnitude_t) + 1e-8)
        magnitude_r = magnitude_r / (np.max(magnitude_r) + 1e-8)
        
        # 相関計算
        correlation = np.corrcoef(magnitude_t.flatten(), magnitude_r.flatten())[0, 1]
        return max(0.0, correlation if not np.isnan(correlation) else 0.0)
    
    def _binary_pattern_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                               region: Tuple[int, int, int, int]) -> float:
        """
        二値パターンによるマッチング（白点分布の類似度）
        """
        x, y, w, h = region
        if x + w > main_img.shape[1] or y + h > main_img.shape[0]:
            return 0.0
        
        roi = main_img[y:y+h, x:x+w]
        
        # 閾値処理で二値化
        _, binary_template = cv2.threshold(template_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 白点の分布パターン比較
        white_points_t = np.sum(binary_template == 255)
        white_points_r = np.sum(binary_roi == 255)
        
        if white_points_t == 0 or white_points_r == 0:
            return 0.0
        
        # 重複する白点の割合
        overlap = np.sum((binary_template == 255) & (binary_roi == 255))
        union = np.sum((binary_template == 255) | (binary_roi == 255))
        
        if union == 0:
            return 0.0
        
        jaccard_index = overlap / union
        return jaccard_index
    
    def _feature_distribution_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                                     region: Tuple[int, int, int, int]) -> float:
        """
        特徴分布による高精度マッチング
        """
        x, y, w, h = region
        if x + w > main_img.shape[1] or y + h > main_img.shape[0]:
            return 0.0
        
        roi = main_img[y:y+h, x:x+w]
        
        # ガウシアンフィルタで平滑化
        template_smooth = gaussian_filter(template_img.astype(float), sigma=1.0)
        roi_smooth = gaussian_filter(roi.astype(float), sigma=1.0)
        
        # 統計的特徴量を計算
        features_t = [
            np.mean(template_smooth),
            np.std(template_smooth),
            np.median(template_smooth),
            np.percentile(template_smooth, 25),
            np.percentile(template_smooth, 75)
        ]
        
        features_r = [
            np.mean(roi_smooth),
            np.std(roi_smooth),
            np.median(roi_smooth),
            np.percentile(roi_smooth, 25),
            np.percentile(roi_smooth, 75)
        ]
        
        # 特徴量の類似度
        correlation = np.corrcoef(features_t, features_r)[0, 1]
        return max(0.0, correlation if not np.isnan(correlation) else 0.0)
    
    def advanced_template_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                                 method: str = 'FEATURE_DISTRIBUTION', 
                                 step_size: int = 1) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        高精度テンプレートマッチング
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            method: マッチング手法
            step_size: スライディングウィンドウのステップサイズ
            
        Returns:
            Tuple[np.ndarray, Tuple[int, int], float]: (結果マップ, 最適位置, 最適値)
        """
        if method not in self.advanced_methods:
            raise ValueError(f"サポートされていない高精度メソッド: {method}")
        
        h, w = template_img.shape
        result_h = (main_img.shape[0] - h) // step_size + 1
        result_w = (main_img.shape[1] - w) // step_size + 1
        
        result = np.zeros((result_h, result_w), dtype=np.float32)
        match_func = self.advanced_methods[method]
        
        best_val = 0.0
        best_loc = (0, 0)
        
        for i in range(result_h):
            for j in range(result_w):
                x = j * step_size
                y = i * step_size
                
                score = match_func(main_img, template_img, (x, y, w, h))
                result[i, j] = score
                
                if score > best_val:
                    best_val = score
                    best_loc = (x, y)
        
        return result, best_loc, best_val
    
    def find_multiple_matches(self, main_img: np.ndarray, template_img: np.ndarray, 
                            method: str = 'FEATURE_DISTRIBUTION', threshold: float = 0.7,
                            use_advanced: bool = True, step_size: int = 1) -> List[Tuple[int, int, float]]:
        """
        複数のマッチング位置を検出（高精度版）
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            method: マッチング手法
            threshold: 閾値
            use_advanced: 高精度手法を使用するか
            step_size: スライディングウィンドウのステップサイズ
            
        Returns:
            List[Tuple[int, int, float]]: [(x, y, confidence), ...]
        """
        if use_advanced and method in self.advanced_methods:
            result, _, _ = self.advanced_template_matching(main_img, template_img, method, step_size)
            locations = np.where(result >= threshold)
            confidences = result[locations]
            
            matches = []
            for pt, conf in zip(zip(*locations[::-1]), confidences):
                x = pt[0] * step_size
                y = pt[1] * step_size
                matches.append((x, y, conf))
        else:
            # 従来のOpenCVマッチング
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
        
        # 信頼度順にソート
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # 非最大抑制（重複する検出を除去）
        matches = self._non_max_suppression(matches, template_img.shape, overlap_threshold=0.3)
        
        return matches
    
    def _non_max_suppression(self, matches: List[Tuple[int, int, float]], 
                           template_shape: Tuple[int, int], overlap_threshold: float = 0.3) -> List[Tuple[int, int, float]]:
        """
        非最大抑制による重複除去
        """
        if not matches:
            return matches
        
        h, w = template_shape
        filtered_matches = []
        
        for current_match in matches:
            x1, y1, conf1 = current_match
            is_valid = True
            
            for existing_match in filtered_matches:
                x2, y2, conf2 = existing_match
                
                # 重複の計算
                overlap_x = max(0, min(x1 + w, x2 + w) - max(x1, x2))
                overlap_y = max(0, min(y1 + h, y2 + h) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                total_area = w * h
                
                if overlap_area / total_area > overlap_threshold:
                    is_valid = False
                    break
            
            if is_valid:
                filtered_matches.append(current_match)
        
        return filtered_matches
    
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
    メイン関数 - 高精度マッチングのサンプル実行
    """
    matcher = ImageMatcher()
    
    # 画像パスを指定
    main_image_path = "main_image.png"
    template_image_path = "template_image.png"
    
    try:
        # 画像を読み込み
        main_img, template_img = matcher.load_images(main_image_path, template_image_path)
        
        print(f"全体画像サイズ: {main_img.shape}")
        print(f"テンプレート画像サイズ: {template_img.shape}")
        
        # 利用可能なマッチング手法を表示
        print("\n利用可能な高精度マッチング手法:")
        for method in matcher.advanced_methods.keys():
            print(f"  - {method}")
        
        print("\n利用可能な標準マッチング手法:")
        for method in matcher.template_methods.keys():
            print(f"  - {method}")
        
        # 複数の手法でマッチングを試行
        methods_to_test = [
            ('FEATURE_DISTRIBUTION', 0.6, True),
            ('BINARY_PATTERN', 0.5, True),
            ('HISTOGRAM_CORRELATION', 0.7, True),
            ('TM_CCOEFF_NORMED', 0.7, False)
        ]
        
        for method, threshold, use_advanced in methods_to_test:
            print(f"\n--- {method} (閾値: {threshold}) ---")
            
            if use_advanced:
                matches = matcher.find_multiple_matches(
                    main_img, template_img, method=method, threshold=threshold,
                    use_advanced=True, step_size=2
                )
            else:
                matches = matcher.find_multiple_matches(
                    main_img, template_img, method=method, threshold=threshold,
                    use_advanced=False
                )
            
            print(f"検出されたマッチ数: {len(matches)}")
            
            for i, (x, y, conf) in enumerate(matches[:5]):  # 上位5個まで表示
                print(f"  マッチ {i+1}: 位置({x}, {y}), 信頼度: {conf:.3f}")
            
            # 最も良い結果を可視化（最初の手法のみ）
            if method == methods_to_test[0][0] and matches:
                print(f"\n{method}の結果を可視化中...")
                matcher.visualize_results(main_img, template_img, matches)
        
        # ユーザーが特定の手法を選択できるオプション
        print("\n特定の手法でマッチングを実行する場合:")
        print("matcher.find_multiple_matches(main_img, template_img,")
        print("    method='FEATURE_DISTRIBUTION', threshold=0.6, use_advanced=True)")
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("サンプル画像を作成します...")
        create_sample_images()
        
        # サンプル画像を作成後、再度実行
        print("\nサンプル画像でマッチングを実行中...")
        main_img, template_img = matcher.load_images(main_image_path, template_image_path)
        
        # 高精度マッチングでテスト
        matches = matcher.find_multiple_matches(
            main_img, template_img, method='FEATURE_DISTRIBUTION', 
            threshold=0.5, use_advanced=True, step_size=2
        )
        
        print(f"検出されたマッチ数: {len(matches)}")
        for i, (x, y, conf) in enumerate(matches):
            print(f"マッチ {i+1}: 位置({x}, {y}), 信頼度: {conf:.3f}")
        
        if matches:
            matcher.visualize_results(main_img, template_img, matches)

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
