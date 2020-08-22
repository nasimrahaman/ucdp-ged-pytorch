from typing import List
from datetime import datetime

import random
import numpy as np
import torch

from ucdp_ged import constants as C


class Transform(object):
    """Base class for a general transform."""

    def apply(self, sample: dict) -> dict:
        """
        This function can be overwritten.

        Parameters
        ----------
        sample : dict
            A dictionary representing the sample. Can contain arbitrary keys
            and values, i.e. they need not be pytorch tensors.

        Returns
        -------
        dict
            The transformed sample dictionary.
        """
        return sample

    def __call__(self, sample: dict) -> dict:
        """Don't overwrite this if you don't have to."""
        sample = dict(sample)
        return self.apply(sample)


class Compose(Transform):
    """Chains together multiple transforms."""

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
    """
    Converts fields in the sample dictionary to pytorch tensors if the
    values are of type:
        * int
        * float
        * numpy.ndarray.
    """

    def apply(self, sample: dict) -> dict:
        for key in sample:
            if isinstance(
                sample[key],
                (
                    int,
                    np.int,
                    np.int32,
                    np.int64,
                    float,
                    np.float,
                    np.float32,
                    np.float64,
                ),
            ):
                sample[key] = torch.tensor(sample[key])
            elif isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])
            else:
                continue
        return sample


# ---------------------------------
# ---------- DATE & TIME ----------
# ---------------------------------


class DateToTimestamp(Transform):
    """
    Converts the date objects (specified in the `KEYS` class attribute)
    to UNIX timestamps (i.e. seconds since 1970).

    Reads
    -----
    date_start : str
    date_end : str

    Writes
    ------
    date_start : float
    date_end : float
    """

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
    """
    Converts the date objects (specified in the `KEYS` class attribute) to the
    number of days since the `TIME_ORIGIN`, as set in `constants.py`.

    Reads
    -----
    date_start : str
    date_end : str

    Writes
    ------
    date_start : int
    date_end : int
    """

    ORIGIN = datetime.strptime(C.TIME_ORIGIN, "%Y-%m-%d %H:%M:%S.%f")

    def apply(self, sample: dict) -> dict:
        for key in self.keys:
            date = datetime.strptime(sample[key], "%Y-%m-%d %H:%M:%S.%f")
            sample[key] = (date - self.ORIGIN).days
        return sample


class TimeStartEndToMidpoint(Transform):
    """
    Computes the temporal mid-point of the conflict from DATE_START and DATE_END
    as `date_mid`. Also computes a quantity `date_delta` such that
    `2 * date_delta` gives the estimated duration of the conflict.

    Reads
    -----
    date_start : str
    date_end : str

    Writes
    ------
    date_mid : float
    date_delta : float
    """

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
    """
    Converts Latitude and Longitude to the n-Vector representation.

    Reads
    -----
    latitude : float
    longitude : float

    Writes
    ------
    n_vector : numpy.ndarray
    """

    def apply(self, sample: dict) -> dict:
        lat, lon = np.deg2rad(sample["latitude"]), np.deg2rad(sample["longitude"])
        sample["n_vector"] = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        return sample


class NormalizeLatLon(Transform):
    """
    Converts Latitude to a scalar between 0 and 1, and longitude to
    one between 0 and 2. Each degree (of lat or lon) corresponds to
    1/180 units after normalization.

    Reads
    -----
    latitude : float
    longitude : float

    Writes
    ------
    normed_latitude : float
    normed_longitude: float
    """

    def apply(self, sample: dict) -> dict:
        lat, lon = sample["latitude"], sample["longitude"]
        # lat runs from +90 to -90; normalize between [0, 1]
        lat = (lat + 90) / 180
        # lon runs from -180 to +180; normalize between [0, 2]
        lon = (lon + 180) / 180
        # Done
        sample["normed_latitude"] = lat
        sample["normed_longitude"] = lon
        return sample


class WherePrecToSpatialDeltaDot(Transform):
    """
    Represents the spatial precision of a given event as a positive scalar,
    which gives the maximum absolute dot product any possible n-Vector can have
    with the spatial mid-point of the conflict.

    To do this, we must assume a mapping from the precision value stated in the
    code-book to the radius of the spatial circle where the event could have happened.

    Reads
    -----
    where_prec : int

    Writes
    ------
    n_vector_delta_dot : float
    """

    # fmt: off
    RADIUS_OF_EARTH = 6371  # KM
    WHERE_PREC_TO_ARCLEN_MAPPING = {  # KM
        1: 1,
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


class NormalizeWherePrec(WherePrecToSpatialDeltaDot):
    """
    Represents the spatial precision of a given event as a positive quantity
    that matches the units in `NormalizeLatLon`. In other words, the radius
    of the circle inside which the event happened is converted to the
    """

    def apply(self, sample: dict) -> dict:
        arc_len = self.WHERE_PREC_TO_ARCLEN_MAPPING[sample["where_prec"]]
        arc_angle = np.rad2deg(arc_len / self.RADIUS_OF_EARTH)
        sample["normed_where_prec"] = arc_angle / 180
        return sample


# ---------------------------------
# -------------- NLP --------------
# ---------------------------------


class PruneAndSepSources(Transform):
    """
    Splits the sources by a SEP token after removing duplicates.

    Optionally, if `keep_num_sources` is specified, samples as many sources
    while discarding the rest (can be safely set to 20).

    Reads
    -----
    source_article : str

    Writes
    ------
    source_article : str
    """

    SEP_TOKEN = "[SEP]"

    def __init__(self, keep_num_sources=None):
        self.keep_num_sources = keep_num_sources

    def apply(self, sample: dict) -> dict:
        sources = set(str(sample["source_article"]).split(";"))
        if self.keep_num_sources is not None and len(sources) > self.keep_num_sources:
            sources = random.sample(sources, self.keep_num_sources)
        sample["source_article"] = self.SEP_TOKEN.join(sources)
        return sample


# ---------------------------------
# --------- RE-LABELING -----------
# ---------------------------------


class RemapIDs(Transform):
    """
    Relabels the ID's of actors, dyads and conflicts such that they are
    contiguous and compatible with pytorch's Embedding module.

    Reads
    -----
    side_a_new_id : int
    side_b_new_id : int
    dyad_new_id: int
    conflict_new_id: int

    Writes
    ------
    side_a_emb_id : int
    side_b_emb_id : int
    dyad_emb_id: int
    conflict_emb_id: int
    """

    def apply(self, sample: dict) -> dict:
        # fmt: off
        sample["side_a_emb_id"] = C.UNIQUE_ACTOR_IDS.index(sample["side_a_new_id"])
        sample["side_b_emb_id"] = C.UNIQUE_ACTOR_IDS.index(sample["side_b_new_id"])
        sample["dyad_emb_id"] = C.UNIQUE_DYAD_IDS.index(sample["dyad_new_id"])
        sample["conflict_emb_id"] = C.UNIQUE_CONFLICT_IDS.index(sample["conflict_new_id"])
        # fmt: on
        return sample


class RemapCategories(Transform):
    """
    Relabels the categories of the specified key to be contiguous.
    For a list of available keys, please refer to the keys of the
    dictionary defined in `ucdp_ged.constants.CATEGORICAL_VARIABLES`.
    """

    def __init__(self, keys=None):
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = list(C.CATEGORICAL_VARIABLES.keys()) if keys is None else keys

    def apply(self, sample: dict) -> dict:
        for key in self.keys:
            sample[key] = C.CATEGORICAL_VARIABLES[key].index(sample[key])
        return sample


# ---------------------------------
# --------- CORRECTIONS -----------
# ---------------------------------


class ComputeTrueDeathCount(Transform):
    """
    The way civilians deaths are handled in UCDP-GED is somewhat peculiar, in that
    civilian deaths are not listed under `deaths_b` even though `side_b_new_id`
    corresponds to that of civilians. Instead, they are reported under
    `deaths_civilians`. This transform corrects for that.

    Reads
    -----
    deaths_a : int
    deaths_b : int

    Writes
    ------
    deaths_a_true : int
    deaths_b_true : int
    """

    def apply(self, sample: dict) -> dict:
        # Note that civilians can never be side_a, so deaths_a_true = deaths_a`.
        sample["deaths_a_true"] = sample["deaths_a"]
        sample["deaths_b_true"] = (
            sample["deaths_civilians"]
            if sample["side_b_new_id"] == C.CIVILIAN_ACTOR_ID
            else sample["deaths_b"]
        )
        return sample
