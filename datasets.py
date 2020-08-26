import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils import data


class LibriSpeech(data.Dataset):
  def __init__(self, home, partition, sampling=1.):
    """
    home: a str indicating path to the LibriSpeech home directory.
    partition: a list containing the partitions to load, e.g.,
      ['train-clean-100', 'train-other-500'].
    sampling: a float indicating the portion of the entire data to take.
    """
    # A dictionary of utterance ids with full path mapped to their lengths.
    # E.g., {'/path/to/librispeech/train-other-500/xxx.pt: 87',
    #        '/path/to/librispeech/train-clean-100/yyy.pt: 6666'}
    self.uid2len = {}
    for p in partition:
      # E.g., split_dir = '/path/to/librispeech/train-other-500'
      split_dir = os.path.join(home, p)
      with open(os.path.join(split_dir, 'lengths.pkl'), 'rb') as f:
        split_uid2len = pickle.load(f)
        self.uid2len.update(
          {os.path.join(split_dir, u): l for u, l in split_uid2len.items()})

    # List of utterance ids with full path.
    self.uids = list(self.uid2len.keys())

    # Sub-sample and update self.uid2len as well.
    self.uids = self.uids[:int(len(self.uids) * sampling)]
    self.uid2len = {u: self.uid2len[u] for u in self.uids}

  def __len__(self):
    return len(self.uids)

  def __getitem__(self, index):
    x = torch.load(self.uids[index])
    l = self.uid2len[self.uids[index]]
    return x, l
