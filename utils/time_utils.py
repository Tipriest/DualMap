import os
import csv
import logging
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from pympler import asizeof
from tabulate import tabulate

# Set up the module-level logger
logger = logging.getLogger(__name__)

def get_timestamp_string():
    """Generate a timestamp string for file/directory naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_timestamped_path(base_path: str, prefix: str = "", suffix: str = "") -> str:
    """
    Generate a timestamped path for saving results.
    
    Args:
        base_path: Base directory path
        prefix: Optional prefix for the timestamp directory
        suffix: Optional suffix for the timestamp directory
    
    Returns:
        Timestamped path string
    """
    timestamp = get_timestamp_string()
    if prefix and suffix:
        timestamped_dir = f"{prefix}_{timestamp}_{suffix}"
    elif prefix:
        timestamped_dir = f"{prefix}_{timestamp}"
    elif suffix:
        timestamped_dir = f"{timestamp}_{suffix}"
    else:
        timestamped_dir = timestamp
    
    return os.path.join(base_path, timestamped_dir)




@contextmanager
def measure_time_block(description=""):
    """
    一个上下文管理器，用于测量代码块的执行时间。
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        message = f"{description}: {elapsed_time:.4f} seconds"

        # Check if logger is configured
        if logging.getLogger().hasHandlers():
            logger.info(message)
        else:
            print(message)


def measure_time(func):
    """
    一个装饰器函数，用于测量另一个函数的执行时间。

    参数:
    func (function): 需要被测量的函数。

    返回:
    function: 测量执行时间的装饰后函数

    Example:
    >>> @measure_time
    >>> def my_function():
    >>>     # Code to be measured
    >>>     pass
    >>> my_function()
    Done! Execution time of my_function function: 0.01 seconds
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        # Call the function with any arguments it was called with
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper


@contextmanager
def timing_context(name, instance, results_attr_name="timing_results"):
    """
    Context manager for measuring execution time with optional storage.
    用于测量执行时间并可选存储的上下文管理器

    Args:
        name (str): 被测量块的名称
        instance (object): 用于存储计时结果的对象
        results_attr_name (str): 实例上用于存储计时结果的属性名称
    """
    # initialize the dict
    if not hasattr(instance, results_attr_name):
        setattr(instance, results_attr_name, {})

    # get dict
    results_dict = getattr(instance, results_attr_name)

    # 使用给定名称初始化列表
    if name not in results_dict:
        results_dict[name] = []

    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time

    # store the elapsed time in the dict
    results_dict[name].append(elapsed_time)


def print_timing_results(label, timing_results):
    # 打印计时结果的表格
    if not timing_results:
        return
    rows = []
    for key, times in timing_results.items():
        avg_time = np.mean(times)
        percentile_90 = np.percentile(times, 90)
        rows.append([key, f"{avg_time:.4f}", f"{percentile_90:.4f}"])
    logger.info(f"\n{label} Timing Results:")
    logger.info(
        tabulate(
            rows,
            headers=["Step", "Avg Time (s)", "90th Percentile (s)"],
            tablefmt="grid",
        )
    )


def save_timing_results(timing_results, csv_file):
    """将计时结果保存到CSV文件。"""
    if not timing_results:
        return

    rows = []
    for key, times in timing_results.items():
        avg_time = np.mean(times)
        percentile_90 = np.percentile(times, 90)
        rows.append([key, f"{avg_time:.4f}", f"{percentile_90:.4f}"])

    # Save results to CSV if a filename is provided
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Avg Time (s)", "90th Percentile (s)"])
        for row in rows:
            writer.writerow(row)
    logger.info(f"\nTiming results saved to {csv_file}")


# Utility function to get memory usage of local and global maps
def get_map_memory_usage(local_map, global_map):
    local_mb = asizeof.asizeof(local_map) / 1024 / 1024
    global_mb = asizeof.asizeof(global_map) / 1024 / 1024
    return {"local_map_mb": round(local_mb, 4), "global_map_mb": round(global_mb, 4)}
