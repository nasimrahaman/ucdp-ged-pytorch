from typing import List
from datetime import datetime

import torch

try:
    from transformers import BertTokenizer
except ImportError:
    BertTokenizer = None
    pass


class Transform(object):
    def apply(self, sample: dict) -> dict:
        return sample

    def __call__(self, sample: dict) -> dict:
        sample = dict(sample)
        return self.apply(sample)


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = list(transforms)

    def apply(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


# ---------------------------------
# ------------ CASTING ------------
# ---------------------------------


class AsTensor(Transform):
    def apply(self, sample: dict) -> dict:
        for key in sample:
            if torch.is_tensor(sample[key]):
                continue
            if not isinstance(sample[key], (int, float)):
                continue
            sample[key] = torch.tensor(sample[key])
        return sample


# ---------------------------------
# ---------- DATE & TIME ----------
# ---------------------------------


class DateToTimestamp(Transform):
    KEYS = {"date_start", "date_end"}

    def __init__(self, keys=None):
        self.keys = keys or self.KEYS

    def apply(self, sample: dict) -> dict:
        for key in self.keys:
            sample[key] = torch.tensor(
                datetime.strptime(sample[key], "%Y-%m-%d %H:%M:%S.%f").timestamp()
            )
        return sample


# ---------------------------------
# ------------ SOURCES ------------
# ---------------------------------


class SplitSources(Transform):
    def apply(self, sample: dict) -> dict:
        sample["source_article"] = sample["source_article"].split(";")
        return sample


class SourcesToBertTokens(Transform):
    def __init__(self, pad_to=128):
        assert BertTokenizer is not None
        self.pad_to = pad_to
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

    def apply(self, sample: dict) -> dict:
        assert isinstance(sample["source_article"], list)
        tokenized_sources = []
        for source in sample["source_article"]:
            tokenized_sources.append(
                self.tokenizer.encode_plus(
                    text=source,
                    add_special_tokens=True,
                    max_length=self.pad_to,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
            )
        sample["source_tokens"] = tokenized_sources
        return sample
