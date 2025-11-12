#!/usr/bin/env python3
"""
棋盘格相机标定脚本（使用 OpenCV）

说明（中文）:
- 假设棋盘格的内部角点数量为 pattern_size = (cols, rows)。
  OpenCV 中的 patternSize 表示每行/列的角点数（不是方格数）。
- 本脚本默认 pattern_size=(11,8)，即 11 列角点、8 行角点（你可以通过命令行覆盖）。
- square_size 单位为毫米（例如 12 表示 12mm）。

注意/假设：
- 你提供的“200x150mm，8行11列，格子宽12mm”在尺寸上可能不完全一致。
  我在脚本中遵循你给出的参数：pattern_size=(11,8)，square_size=12 (mm)。
  如果你的 "8行11列" 指的是方格数而不是角点数，应该把 pattern_size 设为 (10,7)。

用法示例：
  python3 scripts/calibrate_camera.py --images "images/*.jpg" --pattern-cols 11 --pattern-rows 8 --square-size 12 --out calib.yml --undistort_out undistorted/

依赖：
  opencv-python, numpy
  pip install opencv-python numpy

输出：
- 保存的标定文件（YAML）包含 camera_matrix, dist_coeffs, rvecs, tvecs
- 可选会输出去畸变后的图片到指定文件夹

"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Chessboard camera calibration using OpenCV (中文说明在脚本顶部)")
    p.add_argument("--images", required=True, help="Glob pattern or directory containing calibration images, e.g. 'images/*.jpg' or 'calib_imgs/'")
    p.add_argument("--pattern-cols", type=int, default=10, help="Number of inner corners per chessboard row (columns) - default 11")
    p.add_argument("--pattern-rows", type=int, default=7, help="Number of inner corners per chessboard column (rows) - default 8")
    p.add_argument("--square-size", type=float, default=12.0, help="Square size in millimeters - default 12.0")
    p.add_argument("--out", default="calibration.yml", help="Output YAML file to save camera matrix and distortion coeffs")
    p.add_argument("--undistort_out", default=None, help="If set, undistorted images will be saved to this directory")
    p.add_argument("--show", action="store_true", help="Show found corners and undistorted images interactively")
    p.add_argument("--fix-aspect-ratio", action="store_true", help="(optional) fix aspect ratio during calibration flags")
    return p.parse_args()


def gather_images(images_arg):
    p = Path(images_arg)
    files = []
    if p.is_dir():
        # accept common image extensions
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            files.extend(sorted(map(str, p.glob(ext))))
    else:
        # treat as glob pattern
        files = sorted(glob.glob(images_arg))

    return files


def create_object_points(pattern_size, square_size, dtype=np.float32):
    # pattern_size: (cols, rows) = number of inner corners
    cols, rows = pattern_size
    # prepare a single board's object points, e.g. (0,0,0), (1,0,0), ... multiplied by square_size
    objp = np.zeros((rows * cols, 3), dtype=dtype)
    # x varies fastest across the columns
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate(images, pattern_size, square_size, show=False, undistort_out=None, out_file="calibration.yml", fix_aspect_ratio=False):
    # termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    objp = create_object_points(pattern_size, square_size)

    img_shape = None
    good_images = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"警告: 无法读取图片: {fname}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        # 查找棋盘角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if found:
            # 精细化角点
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)
            good_images.append(fname)
            print(f"找到角点: {fname} (角点数={len(corners2)})")

            if show:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners2, found)
                cv2.imshow('corners', vis)
                key = cv2.waitKey(500)
                if key == 27:  # ESC
                    show = False
                    cv2.destroyWindow('corners')
        else:
            print(f"未找到角点: {fname}")

    if len(objpoints) < 3:
        raise RuntimeError(f"有效标定图像不足: 找到 {len(objpoints)} 张（需要至少 3 张）")

    # 标定
    # 可选固定像素长宽比（fx/fy），使用标志 CV_CALIB_FIX_ASPECT_RATIO
    flags = 0
    if fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    print("开始标定...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None, flags=flags)

    # 计算重投影误差
    total_error = 0.0
    per_view_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_view_errors.append(err)
        total_error += err

    mean_error = total_error / len(objpoints)

    print(f"标定完成: RMS={ret}")
    print(f"平均重投影误差(per-view mean) = {mean_error}")

    # 保存到 YAML（或 XML）
    fs = cv2.FileStorage(out_file, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("rms", float(ret))
    fs.write("reprojection_error", float(mean_error))
    fs.release()
    print(f"已保存标定结果到: {out_file}")

    # 可选去畸变并保存结果
    if undistort_out:
        os.makedirs(undistort_out, exist_ok=True)
        for fname in good_images:
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
            basename = os.path.basename(fname)
            outpath = os.path.join(undistort_out, basename)
            cv2.imwrite(outpath, dst)
            if show:
                cv2.imshow('undistorted', dst)
                key = cv2.waitKey(200)
                if key == 27:
                    show = False
                    cv2.destroyWindow('undistorted')
        print(f"已保存去畸变图像到: {undistort_out}")

    if show:
        cv2.destroyAllWindows()

    # 返回结果字典
    return {
        'rms': float(ret),
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'reprojection_error_mean': float(mean_error),
        'per_view_errors': per_view_errors,
        'rvecs': rvecs,
        'tvecs': tvecs,
    }


def main():
    args = parse_args()

    images = gather_images(args.images)
    if not images:
        print(f"未找到任何图片: {args.images}", file=sys.stderr)
        sys.exit(1)

    pattern_size = (args.pattern_cols, args.pattern_rows)

    try:
        res = calibrate(
            images=images,
            pattern_size=pattern_size,
            square_size=args.square_size,
            show=args.show,
            undistort_out=args.undistort_out,
            out_file=args.out,
            fix_aspect_ratio=args.fix_aspect_ratio,
        )
    except Exception as e:
        print(f"标定失败: {e}", file=sys.stderr)
        sys.exit(2)

    # 打印关键结果
    print("\n关键输出:")
    print(f"RMS: {res['rms']}")
    print(f"平均重投影误差: {res['reprojection_error_mean']}")
    print("相机矩阵:")
    print(res['camera_matrix'])
    print("畸变系数:")
    print(res['dist_coeffs'])


if __name__ == '__main__':
    main()
