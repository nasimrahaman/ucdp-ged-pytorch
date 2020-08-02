from torch.utils.data.dataloader import DataLoader
from ucdp_ged.ged import GED
from ucdp_ged import transforms as tr


def make_basic_loader(path="data/ged201.csv", **loader_kwargs):
    transforms = tr.Compose([
        # Date and time
        tr.DateToDaysSinceOrigin(),
        tr.TimeStartEndToMidpoint(),
        # Location
        tr.LatLonToNVector(),
        tr.WherePrecToSpatialDeltaDot(),
        # Actors, dyads and conflicts
        tr.RemapIDs(),
        # Meta info
        tr.RemapCategories(keys="type_of_violence"),
        # NLP
        tr.PruneAndSepSources(keep_num_sources=15),
        # Convert to pytorch
        tr.AsTensor()
    ])
    ged_dataset = GED(path, transforms=transforms)
    if not loader_kwargs:
        return ged_dataset
    else:
        ged_loader = DataLoader(ged_dataset, **loader_kwargs)
        return ged_loader


if __name__ == '__main__':
    ged = make_basic_loader("/Users/nrahaman/Python/ucdp-ged-pytorch/data/ged201.csv",
                            batch_size=2)
    batch = next(iter(ged))
    pass