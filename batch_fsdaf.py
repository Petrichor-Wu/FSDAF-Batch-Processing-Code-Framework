# ===========================================================================
# Project:      FSDAF Batch Processing Code (NDVI Example)
# Author:       Jinxin Wu
# Institution:  National Engineering Research Center for Geographic Information System,
#               China University of Geosciences (Wuhan)
# Date:         November 2025
# Description:  This project aims to provide a stable and efficient batch processing
#               framework for the FSDAF (Flexible Spatio-temporal Data Fusion) algorithm.
#               The system has been comprehensively fixed and optimized to address common
#               issues encountered during remote sensing data fusion (such as index
#               out-of-bounds errors, data type inconsistencies, and file I/O conflicts),
#               achieving automated, multi-threaded (or multi-process) matching and fusion
#               of time-series data.
# ===========================================================================
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


class BatchFSDAFProcessor:

    def __init__(self, config_file: str):

        self.config_file = config_file
        self.load_config()
        self.setup_logging()

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            # 验证必要的配置项
            required_keys = [
                'coarse_prediction_dir',  # 预测时间低分辨率影像文件夹
                'fine_first_pair_dir',  # 第一对高分辨率影像文件夹
                'coarse_first_pair_dir',  # 第一对低分辨率影像文件夹
                'temp_dir',  # 临时文件夹
                'output_dir',  # 输出文件夹
                'parameters_file'  # 参数文件
            ]

            for key in required_keys:
                if key not in self.config:
                    raise ValueError(f"配置文件中缺少必要参数: {key}")

            # 设置默认值
            self.max_time_diff = self.config.get('max_time_diff_days', 16)  # 最大时间差（天）
            self.parallel_workers = self.config.get('parallel_workers', 4)  # 并行处理数
            self.overwrite = self.config.get('overwrite', False)  # 是否覆盖已有结果

        except Exception as e:
            raise Exception(f"加载配置文件失败: {str(e)}")

    def setup_logging(self):
        """设置日志系统"""
        log_dir = os.path.dirname(self.config_file)
        log_file = os.path.join(log_dir, f'fsdaf_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """
        从文件名提取日期
        支持格式: yyyymmdd.tif 或包含日期的其他格式
        """
        try:
            # 尝试匹配 yyyymmdd 格式
            match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
            if match:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day)

            # 尝试其他格式
            match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
            if match:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day)

            return None
        except Exception as e:
            self.logger.warning(f"无法从文件名 {filename} 提取日期: {str(e)}")
            return None

    def find_matching_files(self, target_date: datetime, search_dir: str, max_days: int = 16) -> List[str]:
        """
        在指定目录中查找与目标日期最接近的文件

        Args:
            target_date: 目标日期
            search_dir: 搜索目录
            max_days: 最大允许天数差异

        Returns:
            匹配的文件路径列表（按时间接近度排序）
        """
        if not os.path.exists(search_dir):
            self.logger.error(f"目录不存在: {search_dir}")
            return []

        # 获取目录中所有tif文件
        tif_files = [f for f in os.listdir(search_dir) if f.lower().endswith('.tif')]

        file_dates = []
        for file in tif_files:
            file_path = os.path.join(search_dir, file)
            date = self.extract_date_from_filename(file)
            if date:
                days_diff = abs((date - target_date).days)
                if days_diff <= max_days:
                    file_dates.append((file_path, date, days_diff))

        # 按时间差异排序
        file_dates.sort(key=lambda x: x[2])

        return [item[0] for item in file_dates]

    def get_files_for_processing(self) -> List[Dict]:
        """
        获取需要处理的文件组合

        Returns:
            文件组合列表，每个元素包含:
            {
                'coarse_pred': 预测时间低分辨率影像,
                'fine_first': 第一对高分辨率影像,
                'coarse_first': 第一对低分辨率影像,
                'pred_date': 预测日期
            }
        """
        processing_list = []

        # 获取预测时间低分辨率影像
        coarse_pred_dir = self.config['coarse_prediction_dir']
        if not os.path.exists(coarse_pred_dir):
            raise Exception(f"预测时间低分辨率影像目录不存在: {coarse_pred_dir}")

        coarse_pred_files = [f for f in os.listdir(coarse_pred_dir) if f.lower().endswith('.tif')]

        self.logger.info(f"发现 {len(coarse_pred_files)} 个预测时间低分辨率影像")

        for coarse_pred_file in coarse_pred_files:
            coarse_pred_path = os.path.join(coarse_pred_dir, coarse_pred_file)
            pred_date = self.extract_date_from_filename(coarse_pred_file)

            if not pred_date:
                self.logger.warning(f"无法从文件名提取日期: {coarse_pred_file}")
                continue

            # 查找匹配的高分辨率影像
            fine_first_files = self.find_matching_files(
                pred_date,
                self.config['fine_first_pair_dir'],
                self.max_time_diff
            )

            # 查找匹配的低分辨率影像
            coarse_first_files = self.find_matching_files(
                pred_date,
                self.config['coarse_first_pair_dir'],
                self.max_time_diff
            )

            if not fine_first_files:
                self.logger.warning(f"未找到匹配的高分辨率影像: {coarse_pred_file}")
                continue

            if not coarse_first_files:
                self.logger.warning(f"未找到匹配的低分辨率影像: {coarse_pred_file}")
                continue

            # 使用最匹配的文件
            processing_list.append({
                'coarse_pred': coarse_pred_path,
                'fine_first': fine_first_files[0],
                'coarse_first': coarse_first_files[0],
                'pred_date': pred_date
            })

            self.logger.info(f"找到匹配组合: {os.path.basename(coarse_pred_path)} -> "
                             f"{os.path.basename(fine_first_files[0])}, "
                             f"{os.path.basename(coarse_first_files[0])}")

        return processing_list

    def process_single_combination(self, file_combo: Dict) -> bool:

        try:
            coarse_pred = file_combo['coarse_pred']
            fine_first = file_combo['fine_first']
            coarse_first = file_combo['coarse_first']
            pred_date = file_combo['pred_date']

            # 生成输出文件名
            output_filename = f"fsdaf_result_{pred_date.strftime('%Y%m%d')}.tif"
            output_path = os.path.join(self.config['output_dir'], output_filename)

            # 检查是否已存在
            if os.path.exists(output_path) and not self.overwrite:
                self.logger.info(f"输出文件已存在，跳过: {output_filename}")
                return True

            self.logger.info(f"开始处理: {os.path.basename(coarse_pred)}")

            # 调用FSDAF核心处理函数
            success = self.run_fsdaf_core(
                fine_first,  # 第一对高分辨率影像
                coarse_first,  # 第一对低分辨率影像
                coarse_pred,  # 预测时间低分辨率影像
                output_path,  # 输出路径
                self.config['parameters_file'],
                self.config['temp_dir']
            )

            if success:
                self.logger.info(f"处理完成: {output_filename}")
                return True
            else:
                self.logger.error(f"处理失败: {output_filename}")
                return False

        except Exception as e:
            self.logger.error(f"处理文件组合时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def run_fsdaf_core(self, fine_first: str, coarse_first: str, coarse_pred: str,
                       output_path: str, param_file: str, temp_dir: str) -> bool:
        """
        运行FSDAF核心算法
        """
        try:
            self.logger.info("正在执行FSDAF核心算法...")

            import fsdaf_core
            print("加载 fsdaf_core 的路径：", os.path.abspath(fsdaf_core.__file__))

            # 导入并使用修改后的FSDAF核心函数
            from fsdaf_core import process_fsdaf

            # 调用核心处理函数
            success = process_fsdaf(
                fine_first_path=fine_first,
                coarse_first_path=coarse_first,
                coarse_pred_path=coarse_pred,
                output_path=output_path,
                param_file=param_file,
                temp_dir=temp_dir,
                logger=self.logger
            )

            return success

        except Exception as e:
            self.logger.error(f"FSDAF核心算法执行失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def run_batch_processing(self):

        import shutil
        if os.path.exists(self.config['temp_dir']):
            shutil.rmtree(self.config['temp_dir'])
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        """运行批量处理"""
        try:
            self.logger.info("开始FSDAF批量处理")

            # 获取处理列表
            processing_list = self.get_files_for_processing()

            if not processing_list:
                self.logger.warning("没有找到需要处理的文件组合")
                return

            self.logger.info(f"共找到 {len(processing_list)} 个需要处理的文件组合")

            # 创建输出目录
            os.makedirs(self.config['output_dir'], exist_ok=True)
            os.makedirs(self.config['temp_dir'], exist_ok=True)

            # 处理每个组合
            success_count = 0
            total_count = len(processing_list)

            if self.parallel_workers > 1:
                # 并行处理
                with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                    future_to_combo = {
                        executor.submit(self.process_single_combination, combo): combo
                        for combo in processing_list
                    }

                    for future in as_completed(future_to_combo):
                        combo = future_to_combo[future]
                        try:
                            success = future.result()
                            if success:
                                success_count += 1
                        except Exception as e:
                            self.logger.error(f"并行处理时出错: {str(e)}")
            else:
                # 串行处理
                for combo in processing_list:
                    success = self.process_single_combination(combo)
                    if success:
                        success_count += 1

            self.logger.info(f"批量处理完成: {success_count}/{total_count} 成功")

        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}")
            self.logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description='FSDAF批量处理工具')
    parser.add_argument('config', help='配置文件路径')
    args = parser.parse_args()

    try:
        processor = BatchFSDAFProcessor(args.config)
        processor.run_batch_processing()
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()