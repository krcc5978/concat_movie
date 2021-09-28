import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='動画（画像）縦横連結')

parser.add_argument('--input1', help='入力ファイル1')
parser.add_argument('--color_flag1', help='カラー/グレー判別値1', default=True)

parser.add_argument('--input2', help='入力ファイル2')
parser.add_argument('--color_flag2', help='カラー/グレー判別値2', default=True)

parser.add_argument('--out_dir', help='出力ディレクトリ', default='./')
parser.add_argument('--out_movie', help='動画出力場所', default='output.mp4')
parser.add_argument('--scale_factor', help='出力動画のFPSのスケール値', default=1)
parser.add_argument('--concat_type', help='h:横連結、 v:縦連結', default='h')

args = parser.parse_args()


# 2つの画像を横に連結する関数
def concat_image(im_info1, im_info2, concat_type):
    img1 = im_info1[0]  # 1つ目の画像
    img2 = im_info2[0]  # 2つ目の画像
    color_flag1 = im_info1[1]  # 1つ目の画像のカラー/グレー判別値
    color_flag2 = im_info2[1]  # 2つ目の画像のカラー/グレー判別値

    # 1つ目の画像に対しカラーかグレースケールかによって読み込みを変える
    if color_flag1 == 1:
        h1, w1, ch1 = img1.shape[:3]  # 画像のサイズを取得（グレースケール画像は[:2]
    else:
        h1, w1 = img1.shape[:2]

    # 2つ目の画像に対しカラーかグレースケールかによって読み込みを変える
    if color_flag2 == 1:
        h2, w2, ch2 = img2.shape[:3]  # 画像のサイズを取得（グレースケール画像は[:2]
    else:
        h2, w2 = img2.shape[:2]

    # 2つの画像の縦サイズを比較して、大きい方に合わせて一方をリサイズする
    if h1 > h2:  # 1つ目の画像の方が小さい場合
        h1 = h2  # 小さい方を大きい方と同じ縦サイズにする
        w1 = int((h2 / h1) * w2)  # 縦サイズの変化倍率を計算して横サイズを決定する
        img1 = cv2.resize(img1, (w1, h1))  # 画像リサイズ
    else:  # 2つ目の画像の方が小さい場合
        h2 = h1  # 小さい方を大きい方と同じ縦サイズにする
        w2 = int((h1 / h2) * w1)  # 縦サイズの変化倍率を計算して横サイズを決定する
        img2 = cv2.resize(img2, (w2, h2))  # 画像リサイズ

    if concat_type == 'h':
        img = cv2.hconcat([img1, img2])  # 2つの画像を横方向に連結
    else:
        img = cv2.vconcat([img1, img2])
    return img


# 動画を空間方向に連結させる関数
def m_space_hcombine(movie1, movie2, out_dir, out_movie, concat_type, scale_factor):
    os.makedirs(out_dir, exist_ok=True)
    path1 = movie1[0]  # 1つ目の動画のパス
    path2 = movie2[0]  # 2つ目の動画のパス
    color_flag1 = movie1[1]  # 1つ目の動画がカラーかどうか
    color_flag2 = movie2[1]  # 2つ目の動画がカラーかどうか

    # 2つの動画の読み込み
    movie1_obj = cv2.VideoCapture(path1)
    movie2_obj = cv2.VideoCapture(path2)

    # ファイルからフレームを1枚ずつ取得して動画処理後に保存する
    i = 0  # 第1ループ判定用指標
    while True:
        ret1, frame1 = movie1_obj.read()  # 1つ目の動画のフレームを取得
        ret2, frame2 = movie2_obj.read()  # 2つ目の動画のフレームを取得
        check = ret1 or ret2  # 2つのフレームが共に取得できた時だけTrue（論理演算）
        if check:

            if ret1:
                im_info1 = [frame1, color_flag1]  # 画像連結関数への引数1
            else:
                im_info1 = [np.zeros(frame2.shape[:3], dtype='uint8'), color_flag1]  # 画像連結関数への引数1

            if ret2:
                im_info2 = [frame2, color_flag2]  # 画像連結関数への引数2
            else:
                im_info2 = [np.zeros(frame1.shape[:3], dtype='uint8'), color_flag2]  # 画像連結関数への引数1

            frame_mix = concat_image(im_info1, im_info2, concat_type)  # 画像連結関数の実行

            if i == 0:
                # 動画ファイル保存用の設定
                fps = int(movie1_obj.get(cv2.CAP_PROP_FPS))  # 元動画のFPSを取得
                fps_new = int(fps * scale_factor)  # 動画保存時のFPSはスケールファクターをかける
                frame_size = frame_mix.shape[:3]  # 結合したフレームのサイズを得る
                h = frame_size[0]  # フレームの高さサイズを取得
                w = frame_size[1]  # フレームの横サイズを取得
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 動画保存時のfourcc設定（mp4用）
                video = cv2.VideoWriter(f'{out_dir}/{out_movie}', fourcc, fps_new, (w, h))  # 保存動画の仕様
                i = i + 1  # 初期ループ判定用指標を増分
            else:
                pass
            video.write(frame_mix)  # 動画を保存する
        else:
            break

    # 動画オブジェクトの解放
    movie1_obj.release()
    movie2_obj.release()
    return


if __name__ == '__main__':
    m_space_hcombine([args.input1, args.color_flag1], [args.input2, args.color_flag2],
                     args.out_dir, args.out_movie, args.concat_type, args.scale_factor)