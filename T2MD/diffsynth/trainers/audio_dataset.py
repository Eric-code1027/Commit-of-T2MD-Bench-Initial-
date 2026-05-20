import os
import random
import json
import logging
import copy
import pickle
import string
import numpy as np
import re
import librosa

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import torch.distributed as dist
from torch.utils.data import IterableDataset
from ovi.modules.tokenizers import HuggingfaceTokenizer


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True, random_seed=0, pre_epochs=1):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition
        self.random_seed = random_seed
        self.pre_epochs = pre_epochs

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                if self.pre_epochs > 1:
                    data_duplicates = []
                    # pre allocate data index
                    for i in range(self.pre_epochs):
                        random.Random(i + self.random_seed).shuffle(data)
                        data_duplicates.extend(data)
                    data = data_duplicates
                else:
                    random.Random(self.epoch + self.random_seed).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True, random_seed=0, pre_epochs=1):
        self.lists = lists
        self.sampler = DistributedSampler(
                            shuffle, partition, 
                            random_seed=random_seed,
                            pre_epochs=pre_epochs,
                        )

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        print(sampler_info)
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data

class InfinitDataList(IterableDataset):
    def __init__(self, lists, random_seed=0):
        self.lists = lists
        self.seed = random_seed
        self.sampler = DistributedSampler(
                            shuffle=False, partition=True, 
                            random_seed=random_seed,
                            pre_epochs=-1,
                        )
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        print(sampler_info)
        indexes = self.sampler.sample(self.lists)
        while True:
            random.Random(self.epoch + self.seed).shuffle(indexes)
            self.epoch += 1
            print(f'####DATALIST==RANK{dist.get_rank()}###: START EPOCH {self.epoch}')
            for index in indexes:
                # yield dict(src=src)
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data

class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self.source, f, *self.args, **self.kw)

    def set_dict(self, dct):
        self.source.set_dict(dct)


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists

def shuffle_func(data, shuffle_size=1000, online=True):
    buf = []
    if online:
        # full buf
        for sample in data:
            if len(buf) >= shuffle_size:
                break
            buf.append(sample)
        random.shuffle(buf)
        # online shuffle
        for sample in data:
            i = random.randint(0, shuffle_size) # [0, size]
            if i < shuffle_size:
                tmp = buf[i]
                buf[i] = sample
                sample = tmp
            yield sample
    else:
        for sample in data:
            buf.append(sample)
            if len(buf) >= shuffle_size:
                random.shuffle(buf)
                for x in buf:
                    yield x
                buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x

def sort_func(data, sort_size=500, key='wav', dim=-1, online=True):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    if online:
        # full buffer
        for sample in data:
            if len(buf) >= sort_size:
                break
            buf.append(sample)

        sorted_len = 0 # sort length
        for sample in data:
            if sorted_len == 0: # not sorted
                buf.sort(key=lambda x: x[key].size(dim))
                sorted_len = len(buf) # update sort length
            item = buf[sorted_len-1] # fetch last sorted item
            buf[sorted_len-1] = sample # fill in new item to unsorted pos
            sorted_len -= 1
            yield item
    else:
        for sample in data:
            buf.append(sample)
            if len(buf) >= sort_size:
                buf.sort(key=lambda x: x[key].size(dim))
                for x in buf:
                    yield x
                buf = []
    # The sample left over
    buf.sort(key=lambda x: x[key].size(dim))
    for x in buf:
        yield x


class TextProcessor:
    def __init__(self):
        self.full2half_dict_path = os.path.join(os.path.dirname(__file__), "assets", "full2half.dat")
        self.full2half_dict = {}
        with open(self.full2half_dict_path, 'r', encoding='utf-8') as f:
            try:
                for line in f.readlines():
                    full, half = line.strip("\n").split('\t',maxsplit=1)
                    self.full2half_dict[full] = half
            except Exception as e:
                print(f"读取全半角映射文件时出错：{e}")
                import pdb;pdb.set_trace()
                raise

    def full_to_half(self, text):
        """将全角字符转换为半角字符"""
        return ''.join([self.full2half_dict.get(char, char) for char in text])


def read_file_dict(filepath):
    samples = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            samples.append(line)
    return samples

def compute_campplus_feat(wav_input, wav_sr, num_mel_bins=80, sample_rate=16000):
    if wav_sr != sample_rate:
        wav_input = torchaudio.functional.resample(wav_input,
            orig_freq = wav_sr,
            new_freq = sample_rate)
    feature = kaldi.fbank(wav_input, num_mel_bins=num_mel_bins)
    feature = feature - feature.mean(dim=0, keepdim=True)
    return feature


def remove_punctuation_probabilistic(text, probability=0.1):
    if not text:
        return text

    english_punctuation = set(string.punctuation)  # 英文标点
    chinese_punctuation = {'。', '，', '！', '？', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '…', '—','、'}  # 中文标点
    all_punctuation = english_punctuation.union(chinese_punctuation)
    if text[-1] in all_punctuation:
        if random.random() < probability:
            return text[:-1]
    return text 


def parse_json_sample(data, tokenizer_model_path='/root/Ovi/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl'):
    tokenizer = HuggingfaceTokenizer(name=tokenizer_model_path, clean='whitespace')
    text_converter = TextProcessor()
    for file_path in data:
        try:
            samples = read_file_dict(file_path['src'])
            for sample in samples:
                try:
                    if sample.get('duration', -1) > 15: continue # FIXME hardcode
                    key = sample['uuid']
                    if "text_puc_adjust" in sample:
                        if "language" in sample and sample['language'] == "zh":
                            sample['text'] = sample['text_puc_adjust'].strip()
                        else:
                            sample['text'] = sample['text_puc_adjust_tn'].strip()
                        if random.random() < 0.3:
                            sample['text'] = text_converter.full_to_half(sample['text'])

                    if "text_puc_adjust" in sample:
                        sample['text'] = remove_punctuation_probabilistic(sample['text'].strip(), probability=0.1)

                    text_tokens = tokenizer(sample['text'])[0]
                    waveform, utt_sr = librosa.load(sample['source_path'], sr=16000)
                    waveform = torch.from_numpy(waveform).unsqueeze(0)
                    # spec = compute_campplus_feat(waveform, utt_sr)
                    if utt_sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, orig_freq = wav_sr, new_freq = 16000)
                    duration = waveform.shape[-1] / 16000
                    example = dict(key=key, audio=waveform, duration=duration, 
                                prompt=sample['text'], prompt_tokens=text_tokens,
                                audio_path=sample['source_path'])
                    yield example
                except Exception as ex:
                    print(f"Warning, Failed load data in sample {sample} {ex}")
                    continue
        except Exception as e:
            print(f"Warning, Failed load data in file {file_path} {e}]")
            continue


def filter_by_duration(data, max_duration=15, min_duration=1, max_num_text=512):
    # frame rate is 10ms by default
    for sample in data:
        duration = sample['duration']
        if max_duration is not None and duration > max_duration:
            continue
        if min_duration is not None and duration < min_duration:
            continue
        num_text = len(sample['prompt_tokens'])
        # text > 100hz
        if num_text > duration * 100 or num_text > max_num_text:
            continue
        yield sample


def static_batch(data, batch_size=8):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_duration_in_batch=120):
    buf = []
    longest_frames = 0
    for sample in data:
        frames = sample['duration']
        new_sample_frames = frames
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_duration_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch_func(data, batch_type='static', batch_size=16, max_duration_in_batch=120):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_duration_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def collate_fn(data, phone_pad = 0):
    for sample in data:
        assert isinstance(sample, list)

        wav_lens = torch.tensor([x['audio'].size(1) for x in sample], dtype=torch.float32)
        order = torch.argsort(wav_lens, descending=True)

        # batch = [sample[i] for i in order]
        batch_data = {
            'uid': [sample[i]['key'] for i in order],
            'audio': [sample[i]['audio'] for i in order],
            'duration': [sample[i]['duration'] for i in order],
            'audio_length': [sample[i]['audio'].shape[1] for i in order],
            'prompt': [sample[i]['prompt'] for i in order],
            'audio_path': [sample[i]['audio_path'] for i in order],
        }
        yield batch_data

        # sorted_wav_lens = torch.tensor([wav_lens[i] for i in order], dtype=torch.int32)

        # sorted_wav = [sample[i]['wav'] for i in order]
        # sorted_keys = [sample[i]['key'] for i in order]

        # text_lens = torch.tensor([sample[i]['text_tokens'].size(0) for i in order], dtype=torch.int32)
        # sorted_text_tokens = [sample[i]['text_tokens'] for i in order]
        # padded_sorted_text_tokens = pad_sequence(sorted_text_tokens, batch_first=True, padding_value=phone_pad)

        # sorted_specs = [sample[i]['spec'] for i in order]
        # sorted_spec_lens = torch.tensor([sample[i]['spec'].shape[0] for i in order])
        # padded_sorted_specs = pad_sequence(sorted_specs, batch_first=True, padding_value=0)

        # batch_data = {
        #     "fids": sorted_keys,
        #     "text_token": padded_sorted_text_tokens,
        #     "text_lens" : text_lens,
        #     "speech_token" : padded_sorted_audio_tokens,
        #     "target_lens" : sorted_target_lens,
        #     "specs" : padded_sorted_specs,
        #     "spec_lens" : sorted_spec_lens,
        # }
        # yield batch_data


def Dataset(data_list_file, conf, partition=True, seed=0):
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = InfinitDataList(lists, random_seed=seed)

    parse_conf = conf.get("parse_conf", {})
    dataset = Processor(dataset, parse_json_sample, **parse_conf)
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, filter_by_duration, **filter_conf)

    # shuffle & sort
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, shuffle_func, **shuffle_conf)
    use_sort = conf.get('sort', True)
    if use_sort:
        sort_conf = conf.get('sort_conf', {"key" : "wav"})
        dataset = Processor(dataset, sort_func, **sort_conf)
    # batch & collate
    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, batch_func, **batch_conf)
    collate_conf = conf.get('collate_conf', {})
    dataset = Processor(dataset, collate_fn, **collate_conf)
    return dataset


def build_dataset(data_list, config, train=True, **kwargs):
    data_conf = config
    if not train:
        data_conf = copy.deepcopy(config)
        data_conf['shuffle'] = False
        data_conf['sort'] = False
    dataset = Dataset(data_list, data_conf, partition=train, **kwargs)
    return dataset


if __name__ == '__main__':
    data_list = './workspace/tts_data_v3token_1105/all_data_1105//all-meta-jsons.lst'
    conf = {
        'shuffle': False,
        'shuffle_conf': {
            'shuffle_size': 1000,
            'online': True,
        },
        'sort': False,
        'sort_conf': {
            'key': 'audio',
            'sort_size': 500,
            'online': True,
        },
        'batch_conf': {
            'batch_type': 'static', # change
            'batch_size': 16,
        },
    }
    dataset = build_dataset(data_list, conf)
    for data in dataset:
        import pdb; pdb.set_trace()