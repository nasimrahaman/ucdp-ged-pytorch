from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .ged import GED


def fetch_megabatch(
    dataset: "GED", path: str = None, num_workers: int = 0, dump: bool = True
):
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers
    )
    megabatch = next(iter(loader))
    if dump:
        assert path is not None, "Need a path if `dump` is True."
        torch.save(megabatch, path)
    else:
        return megabatch
