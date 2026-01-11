import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt


def find_latest_log(log_dir: Path) -> Optional[Path]:
    logs = sorted(
        log_dir.glob("*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def detect_device(log_path: Path) -> str:
    """
    从 log (YAML 配置部分) 中检测 device 项，例如:
      device: cpu
      device: cuda
    """
    device_re = re.compile(r"^\s*device:\s*(\S+)", re.IGNORECASE)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = device_re.search(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return "unknown"


def detect_yolo_and_ros(log_path: Path) -> Tuple[str, Optional[int]]:
    """
    从 log 中检测:
      - YOLO 模型路径 (优先 [Detector][Init] Loading YOLO model from ...，否则解析 yolo: 下的 model_path)
      - ros_rate: N
    """
    model = None
    ros_rate: Optional[int] = None

    re_yolo_init = re.compile(
        r"\[Detector\]\[Init\]\s+Loading YOLO model from\s+(\S+)"
    )
    re_model_path = re.compile(r"^\s*model_path:\s*(\S+)")
    re_ros_rate = re.compile(r"^\s*ros_rate:\s*(\d+)", re.IGNORECASE)

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            in_yolo_block = False
            for line in f:
                # ros_rate
                if ros_rate is None:
                    m_r = re_ros_rate.search(line)
                    if m_r:
                        ros_rate = int(m_r.group(1))

                # 优先用 [Detector][Init] 的加载日志
                if model is None:
                    m_init = re_yolo_init.search(line)
                    if m_init:
                        model = m_init.group(1)
                        continue

                # 简单解析配置区的 yolo: 块里的 model_path
                if line.lstrip().startswith("yolo:"):
                    in_yolo_block = True
                    continue
                if in_yolo_block:
                    # yolo 块结束
                    if line.startswith(
                        ("sam:", "fastsam:", "clip:", "device:", "use_rerun")
                    ):
                        in_yolo_block = False
                    else:
                        if model is None:
                            m_mp = re_model_path.search(line)
                            if m_mp:
                                model = m_mp.group(1)
    except OSError:
        pass

    if model is None:
        model = "unknown"
    else:
        # 只保留文件名，例如 yoloe-v8l-seg.pt
        model = os.path.basename(model)

    return model, ros_rate


def parse_log(log_path: Path):
    """
    从 log 中解析:
      - frame_idx: [Detector][Layout] Processing frame idx: N
      - det_time:  [Detector][Visualize] Elapsed time: t seconds
      - total_time:[Main] Processing keyframe K took t seconds.
      - 原始/过滤后检测数量: [Detector][Filter] 从 X 个中过滤出 Y 个
    使用最近一次出现的 frame_idx 作为这几种时间/数量的索引。
    """
    layout_re = re.compile(
        r"\[Detector\]\[Layout\]\s+Processing frame idx:\s*(\d+)"
    )
    det_time_re = re.compile(
        r"\[Detector\]\[Visualize\]\s+Elapsed time:\s*([0-9.]+)\s*seconds"
    )
    total_time_re = re.compile(
        r"\[Main\]\s+Processing keyframe\s+(\d+)\s+took\s+([0-9.]+)\s*seconds"
    )
    det_count_re = re.compile(
        r"\[Detector\]\[Filter\]\s*从\s*(\d+)\s*个中过滤出\s*(\d+)\s*个"
    )

    frames: Dict[int, Dict[str, float]] = {}
    current_frame_idx: Optional[int] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = layout_re.search(line)
            if m:
                current_frame_idx = int(m.group(1))
                frames.setdefault(current_frame_idx, {})
                continue

            m = det_time_re.search(line)
            if m and current_frame_idx is not None:
                t = float(m.group(1))
                frames.setdefault(current_frame_idx, {})
                frames[current_frame_idx]["det_time"] = t
                continue

            m = total_time_re.search(line)
            if m and current_frame_idx is not None:
                t = float(m.group(2))
                frames.setdefault(current_frame_idx, {})
                frames[current_frame_idx]["total_time"] = t
                continue

            m = det_count_re.search(line)
            if m and current_frame_idx is not None:
                raw_cnt = int(m.group(1))
                filt_cnt = int(m.group(2))
                frames.setdefault(current_frame_idx, {})
                frames[current_frame_idx]["raw_cnt"] = raw_cnt
                frames[current_frame_idx]["filt_cnt"] = filt_cnt
                continue

    frame_indices = sorted(frames.keys())
    det_times = []
    total_times = []
    raw_counts = []
    filt_counts = []
    valid_det = []
    valid_total = []

    for idx in frame_indices:
        det = frames[idx].get("det_time")
        total = frames[idx].get("total_time")
        raw = frames[idx].get("raw_cnt")
        filt = frames[idx].get("filt_cnt")

        det_times.append(det)
        total_times.append(total)
        raw_counts.append(raw)
        filt_counts.append(filt)

        if det is not None:
            valid_det.append(det)
        if total is not None:
            valid_total.append(total)

    avg_det = sum(valid_det) / len(valid_det) if valid_det else 0.0
    avg_total = sum(valid_total) / len(valid_total) if valid_total else 0.0

    return (
        frame_indices,
        det_times,
        total_times,
        raw_counts,
        filt_counts,
        avg_det,
        avg_total,
    )


def plot_timing(
    frame_indices,
    det_times,
    total_times,
    raw_counts,
    filt_counts,
    avg_det,
    avg_total,
    log_path: Path,
    show: bool,
    device: str,
    model: str,
    ros_rate: Optional[int],
):
    # 两行子图: 上面时间, 下面检测数量
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- 上: 时间曲线 ---
    det_x = [i for i, t in zip(frame_indices, det_times) if t is not None]
    det_y = [t for t in det_times if t is not None]
    total_x = [i for i, t in zip(frame_indices, total_times) if t is not None]
    total_y = [t for t in total_times if t is not None]

    time_lines = []
    time_labels = []

    if total_x:
        (l1,) = ax1.plot(
            total_x,
            total_y,
            label="Total frame time ([Main] Processing keyframe)",
            color="C0",
            linewidth=1.5,
        )
        time_lines.append(l1)
        time_labels.append(l1.get_label())
    if det_x:
        (l2,) = ax1.plot(
            det_x,
            det_y,
            label="[Detector][Visualize] Elapsed time",
            color="C1",
            linewidth=1.5,
        )
        time_lines.append(l2)
        time_labels.append(l2.get_label())

    ax1.set_ylabel("Time (seconds)")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.legend(time_lines, time_labels, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3)

    text = f"Avg total: {avg_total:.3f} s\nAvg visualize: {avg_det:.3f} s"
    ax1.text(
        0.98,
        0.98,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # --- 下: 检测数量曲线 ---
    raw_x = [i for i, c in zip(frame_indices, raw_counts) if c is not None]
    raw_y = [c for c in raw_counts if c is not None]
    filt_x = [i for i, c in zip(frame_indices, filt_counts) if c is not None]
    filt_y = [c for c in filt_counts if c is not None]

    count_lines = []
    count_labels = []

    if raw_x:
        (l3,) = ax2.step(
            raw_x,
            raw_y,
            where="mid",
            label="Raw detections per frame",
            color="C2",
            linewidth=1.0,
        )
        count_lines.append(l3)
        count_labels.append(l3.get_label())
    if filt_x:
        (l4,) = ax2.step(
            filt_x,
            filt_y,
            where="mid",
            label="Filtered detections per frame",
            color="C3",
            linewidth=1.0,
        )
        count_lines.append(l4)
        count_labels.append(l4.get_label())

    ax2.set_xlabel("Processing frame idx ([Detector][Layout])")
    ax2.set_ylabel("Detections per frame")
    if count_lines:
        ax2.legend(count_lines, count_labels, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 在右上角展示总检测数量
    total_raw = sum(c for c in raw_counts if c is not None)
    total_filt = sum(c for c in filt_counts if c is not None)
    text_counts = f"Total raw: {total_raw}\n" f"Total filtered: {total_filt}"
    ax2.text(
        0.98,
        0.98,
        text_counts,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        f"Frame Timing & Detection Count Analysis\n"
        f"{log_path.name}  "
        f"(device: {device}, yolo: {model}, ros_rate: {ros_rate if ros_rate is not None else 'unknown'} Hz)",
        fontsize=12,
    )

    out_dir = log_path.parent
    out_path = out_dir / f"{log_path.stem}_timing.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved timing plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze frame timing from DualMap log."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a specific log file. If omitted, use latest in --log-dir.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory containing log_*.log files (default: output/map_results/log under project root).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window in addition to saving PNG.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else project_root / "output" / "map_results" / "log"
    )

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = find_latest_log(log_dir)

    if not log_path or not log_path.is_file():
        print(f"No log file found. Checked: {log_path or log_dir}")
        return

    print(f"Using log file: {log_path}")

    (
        frame_indices,
        det_times,
        total_times,
        raw_counts,
        filt_counts,
        avg_det,
        avg_total,
    ) = parse_log(log_path)
    if not frame_indices:
        print("No timing information found in log.")
        return

    device = detect_device(log_path)
    print(f"Detected device: {device}")

    model, ros_rate = detect_yolo_and_ros(log_path)
    print(f"Detected YOLO model: {model}")
    print(
        f"Detected ros_rate: {ros_rate if ros_rate is not None else 'unknown'}"
    )

    plot_timing(
        frame_indices,
        det_times,
        total_times,
        raw_counts,
        filt_counts,
        avg_det,
        avg_total,
        log_path,
        args.show,
        device,
        model,
        ros_rate,
    )


if __name__ == "__main__":
    main()
