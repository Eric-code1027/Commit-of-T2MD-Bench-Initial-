from .operators import *
import torch, json, pandas
import warnings
import traceback
from datetime import datetime
import random


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        max_retries=5,
        skip_on_error=True,
        error_log_path=None,
        verbose_errors=True,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.max_retries = max_retries
        self.skip_on_error = skip_on_error
        self.error_log_path = error_log_path
        self.verbose_errors = verbose_errors
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        # Error tracking
        self.error_count = 0
        self.error_samples = []
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def _load_single_item(self, data_id):
        """加载单个数据项，不带重试逻辑"""
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data
    
    def _log_error(self, data_id, error, data_info):
        """记录错误信息"""
        error_msg = {
            "timestamp": datetime.now().isoformat(),
            "data_id": data_id,
            "data_info": str(data_info),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        
        # 记录到内存
        self.error_count += 1
        if len(self.error_samples) < 100:  # 只保留最近100个错误样本
            self.error_samples.append(error_msg)
        
        # 打印警告
        if self.verbose_errors:
            warnings.warn(
                f"数据加载失败 [ID: {data_id}]: {type(error).__name__}: {str(error)}\n"
                f"数据信息: {data_info}"
            )
        
        # 写入日志文件
        if self.error_log_path:
            try:
                with open(self.error_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_msg, ensure_ascii=False) + '\n')
            except Exception as e:
                warnings.warn(f"无法写入错误日志文件: {e}")
    
    def __getitem__(self, data_id):
        """获取数据项，带错误处理和重试机制"""
        if not self.skip_on_error:
            # 如果不跳过错误，直接加载（原有行为）
            return self._load_single_item(data_id)
        
        # 尝试加载当前数据
        for attempt in range(self.max_retries):
            try:
                return self._load_single_item(data_id)
            except Exception as e:
                # 获取数据信息用于日志
                if self.load_from_cache:
                    data_info = self.cached_data[data_id % len(self.cached_data)]
                else:
                    data_info = self.data[data_id % len(self.data)]
                
                # 记录错误
                self._log_error(data_id, e, data_info)
                
                # 如果不是最后一次尝试，切换到下一个数据
                if attempt < self.max_retries - 1:
                    data_id = (data_id + 1) % len(self)
                    if self.verbose_errors:
                        print(f"  -> 尝试加载下一个数据 (ID: {data_id})")
        
        # max_retries 尝试完之后，随机选择 data_id 再尝试 5 次
        if self.verbose_errors:
            print(f"  -> max_retries ({self.max_retries}) 尝试完毕，开始随机选择 data_id")
        
        for random_attempt in range(5):
            try:
                random_data_id = random.randint(0, len(self) - 1)
                if self.verbose_errors:
                    print(f"  -> 随机尝试 {random_attempt + 1}/5 (随机 ID: {random_data_id})")
                return self._load_single_item(random_data_id)
            except Exception as e:
                # 获取数据信息用于日志
                if self.load_from_cache:
                    data_info = self.cached_data[random_data_id % len(self.cached_data)]
                else:
                    data_info = self.data[random_data_id % len(self.data)]
                
                # 记录错误
                self._log_error(random_data_id, e, data_info)
                
                # 如果是最后一次随机尝试也失败，抛出异常
                if random_attempt == 4:
                    raise RuntimeError(
                        f"连续 {self.max_retries} 次顺序尝试和 5 次随机尝试数据加载均失败。最后的错误: {type(e).__name__}: {str(e)}"
                    ) from e

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
    def validate_data_file(self, data_id):
        """验证单个数据文件是否可以正常加载
        
        Args:
            data_id: 数据索引
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            _ = self._load_single_item(data_id)
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}"
    
    def validate_dataset(self, num_samples=None, show_progress=True):
        """验证数据集中的样本
        
        Args:
            num_samples: 要验证的样本数量，None表示验证全部
            show_progress: 是否显示进度
            
        Returns:
            dict: 包含验证结果的字典
        """
        if num_samples is None:
            num_samples = len(self)
        else:
            num_samples = min(num_samples, len(self))
        
        valid_count = 0
        invalid_samples = []
        
        print(f"开始验证数据集，共 {num_samples} 个样本...")
        
        for i in range(num_samples):
            is_valid, error_msg = self.validate_data_file(i)
            if is_valid:
                valid_count += 1
            else:
                invalid_samples.append({"data_id": i, "error": error_msg})
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"已验证: {i + 1}/{num_samples}, 有效: {valid_count}, 无效: {len(invalid_samples)}")
        
        result = {
            "total": num_samples,
            "valid": valid_count,
            "invalid": len(invalid_samples),
            "invalid_samples": invalid_samples,
            "success_rate": valid_count / num_samples if num_samples > 0 else 0,
        }
        
        print(f"\n验证完成:")
        print(f"  总样本数: {result['total']}")
        print(f"  有效样本: {result['valid']}")
        print(f"  无效样本: {result['invalid']}")
        print(f"  成功率: {result['success_rate']:.2%}")
        
        return result
    
    def get_error_statistics(self):
        """获取错误统计信息
        
        Returns:
            dict: 包含错误统计的字典
        """
        error_types = {}
        for error_sample in self.error_samples:
            error_type = error_sample['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": self.error_count,
            "error_samples_recorded": len(self.error_samples),
            "error_types": error_types,
            "recent_errors": self.error_samples[-10:] if self.error_samples else [],
        }
    
    def print_error_summary(self):
        """打印错误摘要"""
        stats = self.get_error_statistics()
        print("\n" + "="*60)
        print("数据加载错误统计")
        print("="*60)
        print(f"总错误数: {stats['total_errors']}")
        print(f"记录的错误样本数: {stats['error_samples_recorded']}")
        
        if stats['error_types']:
            print("\n错误类型分布:")
            for error_type, count in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        if stats['recent_errors']:
            print(f"\n最近的错误（最多显示10个）:")
            for i, error in enumerate(stats['recent_errors'][-10:], 1):
                print(f"\n  [{i}] 数据ID: {error['data_id']}")
                print(f"      时间: {error['timestamp']}")
                print(f"      类型: {error['error_type']}")
                print(f"      信息: {error['error_message'][:100]}")
        print("="*60 + "\n")