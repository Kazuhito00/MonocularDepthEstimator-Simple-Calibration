#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np

import tkinter as tk
import tkinter.simpledialog as simpledialog


def model_load():
    from midas_predictor.midas_predictor import MiDaSPredictor

    model_path = 'midas_predictor/midas_v2_1_small.onnx'
    model_type = 'small'

    midas_predictor = MiDaSPredictor(model_path, model_type)

    def model_run(image):
        result = midas_predictor(image)
        return result

    return model_run


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    args = parser.parse_args()

    return args


def init_window(rgb_window_name, depth_window_name):
    # Tkinter初期化
    root = tk.Tk()
    root.withdraw()

    # OpenCV初期化
    cv.namedWindow(rgb_window_name)
    cv.setMouseCallback(rgb_window_name, mouse_callback)

    cv.namedWindow(depth_window_name)
    cv.setMouseCallback(depth_window_name, mouse_callback)

    return


def mouse_callback(event, x, y, flags, param):
    global mouse_point
    global relative_d_list, absolute_d_list, calibration_p_list
    global depth_map

    mouse_point = [x, y]

    # 左ボタン押下
    if event == cv.EVENT_LBUTTONDOWN:
        input_data = simpledialog.askstring(
            " ",
            "実測値(cm)を入力",
        )
        try:
            absolute_d = int(float(input_data))

            absolute_d_list.append(absolute_d)
            relative_d_list.append(depth_map[y][x])
            calibration_p_list.append([x, y])
        except:
            pass  # 数値以外


def linear_approximation(x, y):
    n = len(x)
    a = ((np.dot(x, y) - y.sum() * x.sum() / n) /
         ((x**2).sum() - x.sum()**2 / n))
    b = (y.sum() - a * x.sum()) / n
    return a, b


def draw_info(
    image,
    depth_map_,
    elapsed_time,
    mouse_point_,
    relative_d_list_,
    absolute_d_list_,
    calibration_p_list_,
    d_scale=None,
    d_shift=None,
):
    image_width, image_height = image.shape[1], image.shape[0]

    # 描画用フレーム作成
    rgb_frame = copy.deepcopy(image)
    depth_frame = copy.deepcopy(depth_map_)

    # 疑似カラー用の値レンジ調整
    depth_max = depth_frame.max()
    depth_frame = ((depth_frame / depth_max) * 255).astype(np.uint8)
    depth_frame = cv.applyColorMap(depth_frame, cv.COLORMAP_TURBO)

    # マウスポインタ上の推論値描画
    if mouse_point_ is not None:
        point_x = mouse_point_[
            0] if mouse_point_[0] < image_width else image_width
        point_y = mouse_point_[
            1] if mouse_point_[1] < image_height else image_height
        point_x = 0 if point_x < 0 else point_x
        point_y = 0 if point_y < 0 else point_y

        display_d = "{0:.1f}".format(depth_map_[point_y][point_x])

        # キャリブレーション済の場合はcm表記で描画
        if d_scale is not None and d_shift is not None:
            display_d = "{0:.1f}".format(
                ((depth_map_[point_y][point_x] * d_scale) + d_shift)) + "cm"

        # RGB画像
        cv.circle(rgb_frame, (point_x, point_y), 3, (0, 255, 0), thickness=1)
        cv.line(rgb_frame, (point_x, point_y), (point_x + 14, point_y - 14),
                (0, 255, 0),
                thickness=1,
                lineType=cv.LINE_8)
        cv.putText(rgb_frame, display_d, (point_x + 15, point_y - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

        # Depth画像
        cv.circle(depth_frame, (point_x, point_y),
                  3, (255, 255, 255),
                  thickness=1)
        cv.line(depth_frame, (point_x, point_y), (point_x + 14, point_y - 14),
                (255, 255, 255),
                thickness=1,
                lineType=cv.LINE_8)
        cv.putText(depth_frame, display_d, (point_x + 15, point_y - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                   cv.LINE_AA)

    # キャリブレーションポイント描画
    for index, calibration_p in enumerate(calibration_p_list_):
        point_x = calibration_p[
            0] if calibration_p[0] < image_width else image_width
        point_y = calibration_p[
            1] if calibration_p[1] < image_height else image_height
        point_x = 0 if point_x < 0 else point_x
        point_y = 0 if point_y < 0 else point_y

        # RGB画像
        cv.circle(rgb_frame, (point_x, point_y), 3, (0, 255, 0), thickness=1)
        cv.line(rgb_frame, (point_x, point_y), (point_x + 14, point_y - 14),
                (0, 255, 0),
                thickness=1,
                lineType=cv.LINE_8)
        cv.putText(
            rgb_frame, "{0:.1f}".format(relative_d_list_[index]) + " : " +
            str(absolute_d_list_[index]) + "cm", (point_x + 15, point_y - 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA)

        # Depth画像
        cv.circle(depth_frame, (point_x, point_y),
                  3, (255, 255, 255),
                  thickness=1)
        cv.line(depth_frame, (point_x, point_y), (point_x + 14, point_y - 14),
                (255, 255, 255),
                thickness=1,
                lineType=cv.LINE_8)
        cv.putText(
            depth_frame, "{0:.1f}".format(relative_d_list_[index]) + " : " +
            str(absolute_d_list_[index]) + "cm", (point_x + 15, point_y - 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    # 推論時間描画
    # RGB画像
    cv.putText(rgb_frame,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
               cv.LINE_AA)
    # Depth画像
    cv.putText(depth_frame,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
               cv.LINE_AA)

    return rgb_frame, depth_frame


def main():
    global mouse_point
    global relative_d_list, absolute_d_list, calibration_p_list
    global depth_map
    mouse_point = None
    relative_d_list, absolute_d_list, calibration_p_list = [], [], []
    depth_map = None

    # コマンドライン引数
    args = get_args()
    device = args.device
    cap_width = args.width
    cap_height = args.height

    # モデルロード
    model = model_load()

    # カメラ準備
    cap = cv.VideoCapture(device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # GUI準備
    rgb_window_name = 'rgb'
    depth_window_name = 'depth'
    init_window(rgb_window_name, depth_window_name)

    while True:
        start_time = time.time()

        # カメラキャプチャ
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        # Depth推定
        result = model(frame)
        depth_map = cv.resize(result, (frame_width, frame_height))

        # 相対距離と絶対距離を最小二乗法を用いて線形近似
        d_scale, d_shift = None, None
        if len(calibration_p_list) >= 2:
            d_scale, d_shift = linear_approximation(np.array(relative_d_list),
                                                    np.array(absolute_d_list))

        elapsed_time = time.time() - start_time

        # 情報描画
        rgb_frame, depth_frame = draw_info(
            frame,
            depth_map,
            elapsed_time,
            mouse_point,
            relative_d_list,
            absolute_d_list,
            calibration_p_list,
            d_scale,
            d_shift,
        )

        cv.imshow(rgb_window_name, rgb_frame)
        cv.imshow(depth_window_name, depth_frame)
        key = cv.waitKey(1)
        if key == 99:  # c
            absolute_d_list.clear()
            relative_d_list.clear()
            calibration_p_list.clear()
        elif key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
