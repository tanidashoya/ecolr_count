from PIL import Image
import cv2
import numpy as np
# import matplotlib.pyplot as plt

#大腸菌数測定用画像のpngとjpegファイルを繰り返し処理して結果を出力させるコードを書く



# 画像読み込み（RGB形式で読み込む）
image = Image.open("D:\\Users\\tanida\\Desktop\\ecolr.png").convert("RGB")
image_np = np.array(image) 

# RGB画像をHSV（色相・彩度・明度）に変換(image_npは各ピクセルのデータをRGB形式（赤・緑・青）で持っているため人の感覚に近いHSV形式（色相・彩度・明度）)
hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

# 青色の範囲を指定（マスク作成）
lower_blue = np.array([90, 50, 50])     #[色相, 彩度, 明度]の大腸菌数のコロニーかの判定基準において、色相の低いほうの下限を定義（彩度・明度はともに50～255とする）
upper_blue = np.array([150, 255, 255])  #           〃　　　　　上限を定義
mask = cv2.inRange(hsv, lower_blue, upper_blue)  #inRangeの引数は全て同じ形式（例えば全てHSV形式のデータ）を想定しないと数値の意味がかみ合わないので全て同じ形式を想定。全てのピクセルデータで繰り返される
# 変数 mask にはHSV画像内のすべてのピクセルについて、lower_blue〜upper_blue の範囲に入るかどうかをチェックして、
# 該当するピクセルは 255（白）、そうでないピクセルは 0（黒） にした2値画像（マスク画像） を返します。



# 輪郭を検出（青色領域の数をカウント）
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blue_dot_count = len(contours)

# 結果を描画（青い点の輪郭を赤く描く）
# output_image = image_np.copy()
# cv2.drawContours(output_image, contours, -1, (255, 0, 0), 1)

# 結果を表示
# plt.figure(figsize=(8, 8))
# plt.imshow(output_image)
# plt.axis("off")
# plt.title(f"青い点の数: {blue_dot_count}")
# plt.show()

# 点の数を出力
print("青い点の数:", blue_dot_count)



#色相について
"""
色相Hueの値と色の対応（OpenCV基準）
OpenCVでは Hue の値が 0〜179 に正規化されており、以下のように色が割り当てられています：

Hue（色相）値	色
0	赤
30	橙
60	黄緑・緑
90	緑
120	青緑
150	青
179	紫・ピンク寄りの赤
"""

#画像RGBのNumPy配列変換
"""
image_np = np.array(image)  は NumPy配列
画像を「ピクセルのRGB値の集まり」として扱う3次元データ

● 構造（3次元配列）
image_np[行][列] = [R, G, B]

次元	 内容	          例
1次元目	 行（縦）	image_np[0] → 1行目
2次元目	 列（横）	image_np[0][1] → 1行目2列目
3次元目	 色（RGB）	[255, 0, 0] → 赤

● RGBとは？
R（赤）・G（緑）・B（青）＝ 各0〜255の値
ピクセル1つ = RGBのリスト（例：[0, 0, 255] は青）

● .shape でサイズ確認

image_np.shape → (高さ, 幅, 3) 【.shapeはNumPy配列の「次元ごとのサイズ（大きさ）」をタプルで返す属性】
例: (720, 1280, 3) → 縦720 × 横1280 のカラー画像

● 例（2行×2列画像）
image_np = [
  [ [255, 0, 0], [0, 255, 0] ],  # 赤, 緑  ←1行目
  [ [0, 0, 255], [255, 255, 0] ] # 青, 黄  ←2行目
]

※「1行分のピクセルの集まり」が1つのリストで、それが複数ネスト（入れ子）されて全体の画像になる

● 使いどころ
特定の色の抽出（マスク処理）

色の変更やフィルター処理

画像の解析や描画の土台になる！

"""