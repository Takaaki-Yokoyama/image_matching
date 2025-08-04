import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import match_template
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class AdvancedImageMatcher:
    """
    より高度な画像マッチング手法を提供するクラス
    """
    
    def __init__(self):
        pass
    
    def preprocess_images(self, main_img: np.ndarray, template_img: np.ndarray, 
                         gaussian_blur: bool = True, histogram_eq: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像の前処理
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            gaussian_blur: ガウシアンブラー適用
            histogram_eq: ヒストグラム平均化適用
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 前処理済み画像
        """
        main_processed = main_img.copy()
        template_processed = template_img.copy()
        
        if gaussian_blur:
            main_processed = cv2.GaussianBlur(main_processed, (3, 3), 0)
            template_processed = cv2.GaussianBlur(template_processed, (3, 3), 0)
        
        if histogram_eq:
            main_processed = cv2.equalizeHist(main_processed)
            template_processed = cv2.equalizeHist(template_processed)
            
        return main_processed, template_processed
    
    def multi_scale_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                           scales: List[float] = [0.8, 0.9, 1.0, 1.1, 1.2]) -> List[Tuple[int, int, float, float]]:
        """
        マルチスケールテンプレートマッチング
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            scales: スケールのリスト
            
        Returns:
            List[Tuple[int, int, float, float]]: [(x, y, confidence, scale), ...]
        """
        matches = []
        
        for scale in scales:
            # テンプレートをスケーリング
            scaled_template = cv2.resize(template_img, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_CUBIC)
            
            # マッチングが可能かチェック
            if scaled_template.shape[0] > main_img.shape[0] or scaled_template.shape[1] > main_img.shape[1]:
                continue
            
            # テンプレートマッチング実行
            result = cv2.matchTemplate(main_img, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            matches.append((max_loc[0], max_loc[1], max_val, scale))
        
        # 信頼度でソート
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def rotation_invariant_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                                  angles: List[float] = None) -> List[Tuple[int, int, float, float]]:
        """
        回転不変マッチング
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            angles: 回転角度のリスト（度）
            
        Returns:
            List[Tuple[int, int, float, float]]: [(x, y, confidence, angle), ...]
        """
        if angles is None:
            angles = range(0, 360, 15)  # 15度刻み
        
        matches = []
        h, w = template_img.shape
        
        for angle in angles:
            # テンプレートを回転
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated_template = cv2.warpAffine(template_img, M, (w, h))
            
            # マッチング実行
            result = cv2.matchTemplate(main_img, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            matches.append((max_loc[0], max_loc[1], max_val, angle))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def edge_based_matching(self, main_img: np.ndarray, template_img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        エッジベースマッチング
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            
        Returns:
            Tuple[np.ndarray, Tuple[int, int], float]: (結果マップ, 位置, 信頼度)
        """
        # Cannyエッジ検出
        main_edges = cv2.Canny(main_img, 50, 150)
        template_edges = cv2.Canny(template_img, 50, 150)
        
        # エッジでマッチング
        result = cv2.matchTemplate(main_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        return result, max_loc, max_val
    
    def feature_based_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                             detector_type: str = 'ORB') -> List[Tuple[int, int, float]]:
        """
        特徴点ベースマッチング（ORB, SIFT, SURF等）
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            detector_type: 特徴点検出器の種類
            
        Returns:
            List[Tuple[int, int, float]]: マッチング結果
        """
        if detector_type == 'ORB':
            detector = cv2.ORB_create()
        elif detector_type == 'SIFT':
            detector = cv2.SIFT_create()
        else:
            raise ValueError(f"サポートされていない検出器: {detector_type}")
        
        # 特徴点とディスクリプタを検出
        kp1, des1 = detector.detectAndCompute(main_img, None)
        kp2, des2 = detector.detectAndCompute(template_img, None)
        
        if des1 is None or des2 is None:
            return []
        
        # マッチング
        if detector_type == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 良いマッチのみを選択
        good_matches = matches[:min(len(matches), 50)]
        
        if len(good_matches) < 4:
            return []
        
        # ホモグラフィ推定
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return []
        
        # テンプレートの角を変換
        h, w = template_img.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # 中心位置を計算
        center_x = int(np.mean(transformed_corners[:, 0, 0]))
        center_y = int(np.mean(transformed_corners[:, 0, 1]))
        confidence = len(good_matches) / len(kp2)  # マッチ率を信頼度とする
        
        return [(center_x, center_y, confidence)]
    
    def ssim_matching(self, main_img: np.ndarray, template_img: np.ndarray, 
                     step_size: int = 5) -> Tuple[Tuple[int, int], float]:
        """
        SSIM（構造類似性指標）を使用したマッチング
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            step_size: スライディングウィンドウのステップサイズ
            
        Returns:
            Tuple[Tuple[int, int], float]: (最適位置, SSIM値)
        """
        h, w = template_img.shape
        best_ssim = -1
        best_loc = (0, 0)
        
        for y in range(0, main_img.shape[0] - h + 1, step_size):
            for x in range(0, main_img.shape[1] - w + 1, step_size):
                window = main_img[y:y+h, x:x+w]
                current_ssim, _ = ssim(template_img, window, full=True)
                
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_loc = (x, y)
        
        return best_loc, best_ssim
    
    def compare_methods(self, main_img: np.ndarray, template_img: np.ndarray) -> dict:
        """
        複数の手法でマッチングを実行し、結果を比較
        
        Args:
            main_img: 全体画像
            template_img: テンプレート画像
            
        Returns:
            dict: 各手法の結果
        """
        results = {}
        
        # 1. 標準テンプレートマッチング
        result = cv2.matchTemplate(main_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        results['Standard Template Matching'] = {'location': max_loc, 'confidence': max_val}
        
        # 2. エッジベースマッチング
        try:
            _, edge_loc, edge_conf = self.edge_based_matching(main_img, template_img)
            results['Edge-based Matching'] = {'location': edge_loc, 'confidence': edge_conf}
        except:
            results['Edge-based Matching'] = {'location': (0, 0), 'confidence': 0}
        
        # 3. マルチスケールマッチング
        try:
            multi_matches = self.multi_scale_matching(main_img, template_img)
            if multi_matches:
                best_multi = multi_matches[0]
                results['Multi-scale Matching'] = {
                    'location': (best_multi[0], best_multi[1]), 
                    'confidence': best_multi[2], 
                    'scale': best_multi[3]
                }
        except:
            results['Multi-scale Matching'] = {'location': (0, 0), 'confidence': 0}
        
        # 4. 特徴点ベースマッチング
        try:
            feature_matches = self.feature_based_matching(main_img, template_img)
            if feature_matches:
                results['Feature-based Matching'] = {
                    'location': (feature_matches[0][0], feature_matches[0][1]), 
                    'confidence': feature_matches[0][2]
                }
        except:
            results['Feature-based Matching'] = {'location': (0, 0), 'confidence': 0}
        
        return results

def demonstrate_advanced_matching():
    """
    高度なマッチング手法のデモンストレーション
    """
    # サンプル画像作成
    main_img = np.random.randint(0, 256, (400, 600), dtype=np.uint8)
    
    # 特徴的なパターンを追加
    cv2.rectangle(main_img, (100, 100), (200, 150), 255, -1)
    cv2.rectangle(main_img, (150, 125), (175, 140), 0, -1)
    cv2.circle(main_img, (300, 200), 50, 128, -1)
    
    # テンプレート作成（少し回転とスケール変更）
    template_img = main_img[110:140, 110:190]
    
    matcher = AdvancedImageMatcher()
    
    print("=== 高度なマッチング手法の比較 ===")
    results = matcher.compare_methods(main_img, template_img)
    
    for method, result in results.items():
        print(f"{method}:")
        print(f"  位置: {result['location']}")
        print(f"  信頼度: {result['confidence']:.3f}")
        if 'scale' in result:
            print(f"  スケール: {result['scale']:.3f}")
        print()

if __name__ == "__main__":
    demonstrate_advanced_matching()
