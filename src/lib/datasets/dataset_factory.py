from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .dataset.korea_structure import KOREA_STRUCTURE
from .dataset.korea_structure_sample import KOREA_STRUCTURE

dataset_factory = {
  'korea_structure': KOREA_STRUCTURE,
  'korea_structure_sample': KOREA_STRUCTURE,
}

_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
