import pandas as pd
from torch.utils.data.dataset import Dataset
from typing import Callable, List


class GED(Dataset):
    """
    Pytorch Dataset object for the Uppsala Conflict Data Program
    Georeferenced Event Dataset, or UCDP-GED in short.
    """

    FIELDS = {
        # Time stamps
        "date_start",
        "date_end",
        "date_prec",
        # Location
        "latitude",
        "longitude",
        "where_prec",
        # Sides, dyads and conflicts
        "side_a_new_id",
        "side_b_new_id",
        "dyad_new_id",
        "conflict_new_id",
        # Event meta
        "event_clarity",
        "active_year",
        "type_of_violence",
        # Deaths
        "deaths_a",
        "deaths_b",
        "deaths_civilians",
        "deaths_unknown",
        "best",
        "high",
        "low",
        # Source (in natural language)
        "source_article",
        "source_original",
    }

    def __init__(
        self,
        path: str = None,
        data_frame: pd.DataFrame = None,
        transforms: Callable = None,
    ):
        self.path = path
        self.data_frame = data_frame
        self.transforms = transforms
        self.build()

    def build(self):
        if self.data_frame is None:
            # fmt: off
            assert self.path is not None, (
                "Path must be provided if no pandas data-frame is provided."
            )
            # fmt: on
        self.data_frame = pd.read_csv(self.path)

    def fetch(self, idx: int) -> dict:
        series = self.data_frame.iloc[idx]
        return {field: series[field] for field in self.FIELDS}

    def get(self, idx: int) -> dict:
        """Fetches the data-sample at index and runs it through the transforms (if any)."""
        sample = self.fetch(idx)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, item):
        return self.get(item)

    @classmethod
    def collate_fn(cls, samples: List[dict]):
        # TODO
        pass
