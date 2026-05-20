import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"图像文件不存在: {data}")
        
        try:
            image = Image.open(data)
            if self.convert_RGB: image = image.convert("RGB")
            if self.convert_RGBA: image = image.convert("RGBA")
            return image
        except Exception as e:
            raise RuntimeError(f"加载图像失败 [{data}]: {type(e).__name__}: {str(e)}") from e


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        try:
            total_frames = int(reader.count_frames())
        except Exception as e:
            raise RuntimeError(f"无法获取视频帧数: {type(e).__name__}: {str(e)}") from e
        
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"视频文件不存在: {data}")
        
        reader = None
        try:
            reader = imageio.get_reader(data)
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                try:
                    frame = reader.get_data(frame_id)
                    frame = Image.fromarray(frame)
                    frame = self.frame_processor(frame)
                    frames.append(frame)
                except Exception as e:
                    raise RuntimeError(f"处理视频第 {frame_id} 帧失败: {type(e).__name__}: {str(e)}") from e
            return frames
        except FileNotFoundError:
            raise
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"加载视频失败 [{data}]: {type(e).__name__}: {str(e)}") from e
        finally:
            if reader is not None:
                try:
                    reader.close()
                except:
                    pass


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        try:
            images = iio.imread(path, mode="RGB")
        except Exception as e:
            raise RuntimeError(f"无法读取GIF文件 [{path}]: {type(e).__name__}: {str(e)}") from e
        
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"GIF文件不存在: {data}")
        
        try:
            num_frames = self.get_num_frames(data)
            frames = []
            images = iio.imread(data, mode="RGB")
            for img in images:
                try:
                    frame = Image.fromarray(img)
                    frame = self.frame_processor(frame)
                    frames.append(frame)
                    if len(frames) >= num_frames:
                        break
                except Exception as e:
                    raise RuntimeError(f"处理GIF帧失败: {type(e).__name__}: {str(e)}") from e
            return frames
        except FileNotFoundError:
            raise
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"加载GIF失败 [{data}]: {type(e).__name__}: {str(e)}") from e


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    
    def __call__(self, data: str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"音频文件不存在: {data}")
        
        try:
            import librosa
            input_audio, sample_rate = librosa.load(data, sr=self.sr)
            return input_audio
        except Exception as e:
            raise RuntimeError(f"加载音频失败 [{data}]: {type(e).__name__}: {str(e)}") from e
