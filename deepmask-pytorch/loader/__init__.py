from .coco_loader import cocoDataset
from .city_loader import cityDataset

__dataset = {'coco': cocoDataset,'city': cityDataset}


def get_loader(name):
    return __dataset[name]


def dataset_names():
    return list(__dataset.keys())
