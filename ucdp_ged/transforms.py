from typing import List
from datetime import datetime

import numpy as np
import torch

try:
    from transformers import BertTokenizer
except ImportError:
    BertTokenizer = None
    pass

from ucdp_ged import constants as C


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


class DateToDaysSinceOrigin(DateToTimestamp):
    ORIGIN = datetime.strptime(C.TIME_ORIGIN, "%Y-%m-%d %H:%M:%S.%f")

    def apply(self, sample: dict) -> dict:
        for key in self.keys:
            date = datetime.strptime(sample[key], "%Y-%m-%d %H:%M:%S.%f")
            sample[key] = (date - self.ORIGIN).days
        return sample


class TimeStartEndToMidpoint(Transform):
    DATE_START = "date_start"
    DATE_END = "date_end"

    def apply(self, sample: dict) -> dict:
        sample["date_mid"] = (sample[self.DATE_START] + sample[self.DATE_END]) / 2
        sample["date_delta"] = (sample[self.DATE_END] - sample[self.DATE_START]) / 2
        return sample


# ---------------------------------
# ----------- GEOGRAPHY -----------
# ---------------------------------


class LatLonToNVector(Transform):
    def apply(self, sample: dict) -> dict:
        lat, lon = np.deg2rad(sample["latitude"]), np.deg2rad(sample["longitude"])
        sample["n_vector"] = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        return sample


class WherePrecToSpatialDeltaDot(Transform):
    # fmt: off
    RADIUS_OF_EARTH = 6371  # KM
    WHERE_PREC_TO_ARCLEN_MAPPING = {  # KM
        1: 0,
        2: 25,      # This is specified in the code-book
        3: 50,
        4: 100,
        5: 250,
        6: 500,     # This corresponds to the radius of the circle
                    # with the same area as the average country.
        7: 1000,
    }
    # fmt: on

    def apply(self, sample: dict) -> dict:
        arc_len = self.WHERE_PREC_TO_ARCLEN_MAPPING[sample["where_prec"]]
        # Recall that for angles in radians:
        #   arc-len = angle * radius
        # Also, the delta-dot is the maximum dot product a n-vector is allowed to
        # have with the center n-vector where the event happened, given that it's
        # known/assumed that the event happened within a certain radius.
        delta_dot = np.cos(arc_len / self.RADIUS_OF_EARTH)
        sample["n_vector_delta_dot"] = delta_dot
        return sample


# ---------------------------------
# -------------- NLP --------------
# ---------------------------------


class SplitSources(Transform):
    def apply(self, sample: dict) -> dict:
        sample["source_article"] = sample["source_article"].split(";")
        return sample


class SourcesToBertTokens(Transform):

    def __init__(self, max_length: int = 128):
        assert BertTokenizer is not None
        self.max_length = max_length
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
                    max_length=self.max_length,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
            )
        sample["source_tokens"] = tokenized_sources
        return sample


# ---------------------------------
# --------- RE-LABELING -----------
# ---------------------------------


class RemapIDs(Transform):
    def apply(self, sample: dict) -> dict:
        # fmt: off
        sample["side_a_emb_id"] = C.UNIQUE_ACTOR_IDS.index(sample["side_a_new_id"])
        sample["side_b_emb_id"] = C.UNIQUE_ACTOR_IDS.index(sample["side_b_new_id"])
        sample["dyad_emb_id"] = C.UNIQUE_DYAD_IDS.index(sample["dyad_new_id"])
        sample["conflict_emb_id"] = C.UNIQUE_CONFLICT_IDS.index(sample["conflict_new_id"])
        # fmt: on
        return sample
