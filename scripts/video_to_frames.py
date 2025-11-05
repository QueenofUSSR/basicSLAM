#!/usr/bin/env python3
"""
Simple utility to extract frames from a video using OpenCV or ffmpeg as a fallback.

Usage examples:
    python scripts/video_to_frames.py input.mp4 -o frames_dir
    python scripts/video_to_frames.py input.mp4 -o frames -k 10   # save every 10th frame
    python scripts/video_to_frames.py input.mov -o frames --use-ffmpeg  # use ffmpeg extraction (safe for some HEVC files)
"""

import os
import argparse
import subprocess


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video file")
    p.add_argument("input", help="Path to input video file")
    p.add_argument("-o", "--output", default="frames", help="Output directory for frames")
    p.add_argument("-f", "--format", default="jpg", choices=["jpg", "png", "jpeg"], help="Image format for saved frames")
    p.add_argument("-s", "--start", type=int, default=0, help="Start frame index (inclusive)")
    p.add_argument("-e", "--end", type=int, default=None, help="End frame index (inclusive). If omitted, goes until video end")
    p.add_argument("-k", "--step", type=int, default=1, help="Save every k-th frame (default: 1 -> every frame)")
    p.add_argument("--zero-pad", type=int, default=6, help="Zero padding width for frame numbers in filenames")
    p.add_argument("--use-ffmpeg", action="store_true", help="Use ffmpeg command-line to extract frames (avoids OpenCV crashes on some codecs)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        raise SystemExit(2)

    os.makedirs(args.output, exist_ok=True)

    # If user requested ffmpeg, use it (avoids importing/using OpenCV)
    if args.use_ffmpeg:
        ok = extract_with_ffmpeg(args.input, args.output, args.format, args.start, args.end, args.step, args.zero_pad)
        if not ok:
            raise SystemExit(4)
        return

    # Otherwise try OpenCV (may crash on some platform/codecs); if you see segfaults,
    # re-run with --use-ffmpeg.
    try:
        import cv2
    except Exception as exc:
        print("Failed to import OpenCV (cv2):", exc)
        print("Try re-running with --use-ffmpeg")
        raise

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Failed to open video: {args.input}")
        raise SystemExit(3)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    print(f"Opened video: {args.input}")
    print(f"Total frames (reported): {total_frames}, FPS: {fps}")
    print(f"Saving to: {args.output}  format: {args.format}  step: {args.step}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # check start/end
        if frame_idx < args.start:
            frame_idx += 1
            continue
        if args.end is not None and frame_idx > args.end:
            break

        if (frame_idx - args.start) % args.step == 0:
            filename = f"frame_{frame_idx:0{args.zero_pad}d}.{args.format}"
            out_path = os.path.join(args.output, filename)
            success = cv2.imwrite(out_path, frame)
            if not success:
                print(f"Warning: failed to write frame {frame_idx} to {out_path}")
            else:
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Frames processed: {frame_idx}, saved: {saved}")


def extract_with_ffmpeg(input_path, output_dir, fmt, start, end, step, zero_pad):
    """Use ffmpeg to extract frames. Returns True on success."""
    # Build output pattern
    pattern = os.path.join(output_dir, f"frame_%0{zero_pad}d.{fmt}")

    # Build select filter depending on start/end/step
    filters = []
    if start and start > 0:
        start_val = int(start)
    else:
        start_val = 0

    if end is not None:
        # ffmpeg's select uses frame numbers starting at 0
        sel_range = f"between(n\,{start_val}\,{int(end)})"
        filters.append(sel_range)

    if step and int(step) > 1:
        filters.append(f"not(mod(n-{start_val}\,{int(step)}))")

    if start_val and not filters:
        # only start provided
        filters.append(f"gte(n\,{start_val})")

    vf = None
    if filters:
        vf = "*".join(filters)

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-vsync", "0", "-q:v", "2", pattern]

    print("Running ffmpeg:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(res.stderr.decode('utf-8', errors='ignore'))
        return True
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:")
        print(e.stderr.decode('utf-8', errors='ignore'))
        return False


if __name__ == "__main__":
    main()
