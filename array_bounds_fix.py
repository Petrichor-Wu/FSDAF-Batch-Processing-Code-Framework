# -*- coding: utf-8 -*-
"""
数组边界检查和修复模块
专门用于修复FSDAF算法中的数组越界问题
"""

import numpy as np
import logging


def safe_index_check(index, array_size, axis_name="", logger=None):
    if index < 0 or index >= array_size:
        if logger:
            logger.error(f"索引越界: {axis_name}索引 {index} 超出范围 [0, {array_size - 1}]")
        return False
    return True


def safe_2d_index_check(row_idx, col_idx, array_shape, logger=None):
    if not safe_index_check(row_idx, array_shape[0], "行", logger):
        return False
    if not safe_index_check(col_idx, array_shape[1], "列", logger):
        return False
    return True


def safe_array_slice(array, row_start, row_end, col_start, col_end, logger=None):
    try:
        # 确保索引在有效范围内
        rows, cols = array.shape[:2]

        row_start = max(0, min(row_start, rows))
        row_end = max(0, min(row_end, rows))
        col_start = max(0, min(col_start, cols))
        col_end = max(0, min(col_end, cols))

        # 检查切片是否有效
        if row_start >= row_end or col_start >= col_end:
            if logger:
                logger.warning(f"无效切片: [{row_start}:{row_end}, {col_start}:{col_end}]")
            return None

        return array[row_start:row_end, col_start:col_end]
    except Exception as e:
        if logger:
            logger.error(f"切片访问错误: {e}")
        return None


def validate_block_coordinates(ind_patch1, ind_patch, location, isub, orig_ns, orig_nl, logger=None):
    try:
        # 检查原始块坐标
        row1, row2 = ind_patch1[isub, 2], ind_patch1[isub, 3]
        col1, col2 = ind_patch1[isub, 0], ind_patch1[isub, 1]

        # 检查是否超出图像范围
        if row2 >= orig_nl or col2 >= orig_ns:
            if logger:
                logger.warning(
                    f"块 {isub + 1} 超出原始图像范围: row2={row2}, col2={col2}, orig_nl={orig_nl}, orig_ns={orig_ns}")
            return False

        # 检查是否为无效矩形
        if row2 < row1 or col2 < col1:
            if logger:
                logger.warning(f"块 {isub + 1} 为无效矩形: [{row1}:{row2}, {col1}:{col2}]")
            return False

        # 检查扩展块坐标
        r1, r2 = ind_patch[isub, 2], ind_patch[isub, 3]
        c1, c2 = ind_patch[isub, 0], ind_patch[isub, 1]

        # 检查扩展块是否有效
        if r2 < r1 or c2 < c1:
            if logger:
                logger.warning(f"块 {isub + 1} 扩展块无效: [{r1}:{r2}, {c1}:{c2}]")
            return False

        # 检查扩展块是否超出边界
        if r2 >= orig_nl or c2 >= orig_ns:
            if logger:
                logger.warning(f"块 {isub + 1} 扩展块超出边界: [{r1}:{r2}, {c1}:{c2}], orig=[{orig_nl},{orig_ns}]")
            return False

        # 检查位置偏移是否有效
        loc = location[isub]
        if loc[1] >= orig_ns or loc[3] >= orig_nl:
            if logger:
                logger.warning(f"块 {isub + 1} 位置偏移越界: location={loc}, orig=[{orig_nl},{orig_ns}]")
            return False

        return True

    except Exception as e:
        if logger:
            logger.error(f"验证块坐标时出错: {e}")
        return False


def fix_index_out_of_bounds_error(data_array, row_indices, col_indices, logger=None):
    try:
        rows, cols = data_array.shape[:2]

        # 修复行索引
        if isinstance(row_indices, np.ndarray):
            row_indices = np.clip(row_indices, 0, rows - 1)
        else:
            row_indices = max(0, min(row_indices, rows - 1))

        # 修复列索引
        if isinstance(col_indices, np.ndarray):
            col_indices = np.clip(col_indices, 0, cols - 1)
        else:
            col_indices = max(0, min(col_indices, cols - 1))

        return row_indices, col_indices

    except Exception as e:
        if logger:
            logger.error(f"修复索引错误时出错: {e}")
        return None, None


def create_safe_window(center_row, center_col, window_size, array_shape, logger=None):
    try:
        rows, cols = array_shape[:2]
        half_window = window_size // 2

        # 计算窗口边界
        row_start = max(0, center_row - half_window)
        row_end = min(rows, center_row + half_window + 1)
        col_start = max(0, center_col - half_window)
        col_end = min(cols, center_col + half_window + 1)

        # 确保窗口至少包含一个像素
        if row_start >= row_end or col_start >= col_end:
            if logger:
                logger.warning(f"无法创建有效窗口: center=({center_row},{center_col}), shape={array_shape}")
            return None, None, None, None

        return row_start, row_end, col_start, col_end

    except Exception as e:
        if logger:
            logger.error(f"创建安全窗口时出错: {e}")
        return None, None, None, None


class ArrayBoundsChecker:

    def __init__(self, logger=None):
        self.logger = logger
        self.error_count = 0
        self.warning_count = 0

    def check_and_fix_bounds(self, array, indices, axis_name=""):
        """检查并修复索引边界"""
        try:
            if isinstance(indices, (list, tuple)):
                fixed_indices = []
                for i, idx in enumerate(indices):
                    if idx < 0 or idx >= array.shape[i]:
                        self.error_count += 1
                        if self.logger:
                            self.logger.error(f"索引越界: {axis_name}[{i}]={idx}, 范围=[0,{array.shape[i] - 1}]")
                        fixed_idx = max(0, min(idx, array.shape[i] - 1))
                        fixed_indices.append(fixed_idx)
                    else:
                        fixed_indices.append(idx)
                return tuple(fixed_indices)
            else:
                if indices < 0 or indices >= array.shape[0]:
                    self.error_count += 1
                    if self.logger:
                        self.logger.error(f"索引越界: {axis_name}={indices}, 范围=[0,{array.shape[0] - 1}]")
                    return max(0, min(indices, array.shape[0] - 1))
                return indices

        except Exception as e:
            if self.logger:
                self.logger.error(f"检查索引边界时出错: {e}")
            self.error_count += 1
            return indices

    def get_stats(self):
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count
        }


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    test_array = np.random.rand(100, 200)
    logger = logging.getLogger(__name__)

    # 测试安全索引检查
    print("测试安全索引检查:")
    print(f"索引 (50, 100): {safe_2d_index_check(50, 100, test_array.shape, logger)}")
    print(f"索引 (150, 250): {safe_2d_index_check(150, 250, test_array.shape, logger)}")

    # 测试安全切片访问
    print("\n测试安全切片访问:")
    slice_data = safe_array_slice(test_array, 10, 20, 30, 40, logger)
    print(f"切片形状: {slice_data.shape if slice_data is not None else 'None'}")

    # 测试安全窗口创建
    print("\n测试安全窗口创建:")
    window = create_safe_window(50, 100, 11, test_array.shape, logger)
    print(f"窗口边界: {window}")