# 高精度画像マッチングシステム

白黒画像のテンプレートマッチングを行い、形状が少し違っていても白点の分布状態が似ていれば検出できる高精度なマッチングシステムです。

## 特徴

- **5つの高精度マッチング手法**を実装
- **柔軟な閾値設定**で検出精度を調整可能
- **形状の違いに対応**した白点分布マッチング
- **非最大抑制**による重複検出の除去
- **可視化機能**でマッチング結果を確認

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使い方

```python
from image_matching import ImageMatcher

matcher = ImageMatcher()

# 画像を読み込み
main_img, template_img = matcher.load_images("main_image.png", "template_image.png")

# 高精度マッチングを実行
matches = matcher.find_multiple_matches(
    main_img, template_img,
    method='FEATURE_DISTRIBUTION',  # 推奨手法
    threshold=0.6,                  # 閾値（0.0-1.0）
    use_advanced=True,              # 高精度手法を使用
    step_size=2                     # 高速化のため
)

# 結果を表示
for i, (x, y, confidence) in enumerate(matches):
    print(f"マッチ {i+1}: 位置({x}, {y}), 信頼度: {confidence:.3f}")

# 結果を可視化
matcher.visualize_results(main_img, template_img, matches)
```

## マッチング手法

### 1. FEATURE_DISTRIBUTION（特徴分布マッチング）
- **用途**: 形状が違っても検出したい場合
- **特徴**: 統計的特徴量による高精度マッチング
- **推奨閾値**: 0.4-0.7
- **適用例**: 手書き文字、ノイズのある画像

```python
matches = matcher.find_multiple_matches(
    main_img, template_img,
    method='FEATURE_DISTRIBUTION',
    threshold=0.6,
    use_advanced=True
)
```

### 2. BINARY_PATTERN（二値パターンマッチング）
- **用途**: 白点の分布パターンを重視
- **特徴**: Jaccard係数による類似度計算
- **推奨閾値**: 0.3-0.6
- **適用例**: 点群データ、スキャン画像

```python
matches = matcher.find_multiple_matches(
    main_img, template_img,
    method='BINARY_PATTERN',
    threshold=0.5,
    use_advanced=True
)
```

### 3. HISTOGRAM_CORRELATION（ヒストグラム相関）
- **用途**: 明度分布の類似性を重視
- **特徴**: ヒストグラム比較による高速マッチング
- **推奨閾値**: 0.6-0.9
- **適用例**: 一般的な画像マッチング

### 4. EDGE_CORRELATION（エッジ相関）
- **用途**: 輪郭パターンを重視
- **特徴**: Sobelエッジ検出を使用
- **推奨閾値**: 0.3-0.6
- **適用例**: 境界線が重要な画像

### 5. GRADIENT_CORRELATION（勾配相関）
- **用途**: 勾配情報による詳細マッチング
- **特徴**: 画像の勾配方向を比較
- **推奨閾値**: 0.4-0.8
- **適用例**: テクスチャ解析

## パラメータ調整

### 閾値（threshold）
- **低い値（0.3-0.5）**: より多くのマッチを検出（偽陽性増加）
- **高い値（0.7-0.9）**: 厳密なマッチのみ検出（偽陽性減少）

### ステップサイズ（step_size）
- **1**: 最高精度（処理時間長）
- **2-4**: バランス型（推奨）
- **5以上**: 高速処理（精度低下）

## 実行例

```bash
# メインプログラム実行
python image_matching.py

# 使用例プログラム実行
python example_usage.py
```

## 結果の解釈

```
検出されたマッチ数: 26
  マッチ 1: 位置(110, 110), 信頼度: 1.000
  マッチ 2: 位置(260, 170), 信頼度: 0.837
  マッチ 3: 位置(255, 191), 信頼度: 0.837
```

- **位置**: テンプレート画像の左上角の座標
- **信頼度**: マッチングの確信度（0.0-1.0）
- 信頼度順に並び、重複は自動的に除去

## トラブルシューティング

### 検出されない場合
1. 閾値を下げる（0.3-0.5）
2. 異なる手法を試す
3. step_sizeを1に設定

### 偽陽性が多い場合
1. 閾値を上げる（0.7-0.9）
2. より厳密な手法を使用
3. 非最大抑制の閾値を調整

### 処理が遅い場合
1. step_sizeを増やす（2-4）
2. 画像サイズを縮小
3. 閾値を上げて早期終了

## ファイル構成

```
image_matching/
├── image_matching.py      # メインクラス
├── example_usage.py       # 使用例
├── requirements.txt       # 依存パッケージ
├── main_image.png        # サンプル全体画像
├── template_image.png    # サンプルテンプレート画像
└── README.md            # このファイル
```
