# :globe_with_meridians: ucdp-ged-pytorch

Batteries-included PyTorch data-loaders for [Uppsala Conflict Data Program](https://ucdp.uu.se/) [Georeferenced Event Dataset](https://ucdp.uu.se/downloads/index.html#ged_global). 

**Disclaimer:** this python package is **not** officially distributed by UCDP. For questions about the dataset, please refer to the official [UCDP homepage](https://www.pcr.uu.se/research/ucdp/). 

## Installation

From source:

```
git clone https://github.com/nasimrahaman/ucdp-ged-pytorch.git
cd ucdp-ged-pytorch
pip install -e .
```

Or with `pip`:

```
pip install git+https://github.com/nasimrahaman/ucdp-ged-pytorch.git
```

## Usage

First, download and unzip the the CSV dataset from the [official download page](https://ucdp.uu.se/downloads/index.html#ged_global).

Now:

```python
from torch.utils.data.dataloader import DataLoader
from ucdp_ged.ged import GED
from ucdp_ged import transforms as tr

transforms = tr.Compose(
        [
            # Normalize date and time
            tr.DateToDaysSinceOrigin(),
            tr.TimeStartEndToMidpoint(),
            # Convert location to n-Vector
            tr.LatLonToNVector(),
            tr.WherePrecToSpatialDeltaDot(),
            # Relabel actors, dyads and conflict IDs to contiguous labels
            tr.RemapIDs(),
            tr.RemapCategories(keys="type_of_violence"),
            # Seperate headlines by a [SEP] token (useful for BERT LMs)
            tr.PruneAndSepSources(keep_num_sources=15),
            # Convert to pytorch tensors
            tr.AsTensor(),
        ]
    )

ged_dataset = GED("path/to/ged201.csv", transforms=transforms)
ged_loader = DataLoader(ged_dataset, batch_size=16, shuffle=True)

for sample in ged_loader: 
    # sample is a dictionary with various keys. 
    ...
```
