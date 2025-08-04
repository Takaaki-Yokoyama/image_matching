# 画像マッチングプロジェクト

白黒画像の一部分から全体画像内での位置を特定するPythonプロジェクトです。

## 機能

### 基本機能
- **テンプレートマッチング**: OpenCVの各種テンプレートマッチング手法
- **複数マッチ検出**: 閾値以上の全ての候補位置を検出
- **結果可視化**: マッチング結果の視覚的表示

### 高度な機能
- **マルチスケールマッチング**: 異なるスケールでのマッチング
- **回転不変マッチング**: 回転した画像でもマッチング可能
- **エッジベースマッチング**: Cannyエッジ検出を使用
- **特徴点ベースマッチング**: ORB、SIFTを使用した特徴点マッチング
- **SSIM マッチング**: 構造類似性指標を使用

## インストール

```bash
# 必要なライブラリをインストール
pip install -r requirements.txt
```

## 使用方法

### 1. 基本的な使用例

```python
from image_matching import ImageMatcher

# マッチャーを初期化
matcher = ImageMatcher()

# 画像を読み込み
main_img, template_img = matcher.load_images("main_image.png", "template_image.png")

# マッチング実行
result, best_loc, best_val = matcher.template_matching(main_img, template_img)
print(f"最適位置: {best_loc}, 信頼度: {best_val}")

# 複数マッチを検出
matches = matcher.find_multiple_matches(main_img, template_img, threshold=0.7)

# 結果を可視化
matcher.visualize_results(main_img, template_img, matches)
```

### 2. 高度な手法を使用

```python
from advanced_matching import AdvancedImageMatcher

matcher = AdvancedImageMatcher()

# マルチスケールマッチング
multi_matches = matcher.multi_scale_matching(main_img, template_img)

# 回転不変マッチング
rotation_matches = matcher.rotation_invariant_matching(main_img, template_img)

# 手法比較
results = matcher.compare_methods(main_img, template_img)
```

### 3. デモ実行

```bash
# 実践例とベンチマークを実行
python demo.py
```

## ファイル構成

- `image_matching.py`: 基本的なマッチング機能
- `advanced_matching.py`: 高度なマッチング手法
- `demo.py`: 実践例とデモンストレーション
- `requirements.txt`: 必要なライブラリ

## マッチング手法の説明

### 1. テンプレートマッチング
OpenCVの`matchTemplate`を使用した標準的な手法：
- `TM_CCOEFF_NORMED`: 正規化相関係数
- `TM_CCORR_NORMED`: 正規化相関
- `TM_SQDIFF_NORMED`: 正規化二乗差

### 2. マルチスケールマッチング
テンプレートを異なるスケールでリサイズしてマッチング。
サイズが異なる画像でも対応可能。

### 3. 回転不変マッチング
テンプレートを様々な角度で回転させてマッチング。
回転した画像でも検出可能。

### 4. エッジベースマッチング
Cannyエッジ検出で輪郭を抽出してからマッチング。
照明変化に強い。

### 5. 特徴点ベースマッチング
ORBやSIFTなどの特徴点を使用。
幾何学変換に対してロバスト。

## パフォーマンス比較

| 手法 | 精度 | 速度 | ロバスト性 |
|------|------|------|------------|
| 標準テンプレート | 高 | 高 | 低 |
| マルチスケール | 高 | 中 | 中 |
| 回転不変 | 中 | 低 | 高 |
| エッジベース | 中 | 高 | 中 |
| 特徴点ベース | 高 | 低 | 高 |

## 使用例のシナリオ

1. **品質検査**: 製品の欠陥検出
2. **医療画像**: X線やMRI画像での異常検出
3. **監視システム**: 特定オブジェクトの検出
4. **画像検索**: 類似画像の検索
5. **ロボットビジョン**: オブジェクトの位置特定

## トラブルシューティング

### よくある問題

1. **画像が読み込めない**
   - ファイルパスが正しいか確認
   - 画像ファイル形式をチェック

2. **マッチング精度が低い**
   - 前処理（ガウシアンブラー、ヒストグラム平均化）を試す
   - 異なるマッチング手法を試す
   - 閾値を調整

3. **処理が遅い**
   - 画像サイズを縮小
   - ステップサイズを大きくする
   - より高速な手法を使用

### パラメータ調整のコツ

- **閾値**: 0.7-0.9が一般的な範囲
- **スケール範囲**: 0.5-2.0で調整
- **回転角度**: 用途に応じて調整（通常は15-30度刻み）

## 拡張性

このプロジェクトは以下のように拡張できます：

1. **新しいマッチング手法の追加**
2. **リアルタイム処理の実装**
3. **GUI インターフェースの追加**
4. **機械学習ベースの手法の統合**

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
