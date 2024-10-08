"""
Parent dataset for sequential data.

Code from https://github.com/HazyResearch/state-spaces/blob/main/src/dataloaders/base.py
"""

from functools import partial
import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from utils.config import is_list

# Default data path is environment variable or hippo/data
if (default_data_path := os.getenv("DATA_PATH")) is None: # := is walrus operator. It is assignment and condition operator at the same time. here it checks the condition after assigning the value to the variable default_data_path
    default_data_path = Path(__file__).parent.parent.parent.absolute()#__file__ is a built in variable it give the location of current script. So traveling through the directory tree is relative to this.
    default_data_path = default_data_path / "dataloaders" / "data" #All the raw datasets should be stored in this directory:- Here is the code containing reading the data
else:
    default_data_path = Path(default_data_path).absolute()

#This SequenceDataset class handled all the primary preprocessing of a data thoroughly.
class SequenceDataset:
    registry = {} #to register the parent classes as dict
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class. All the datasets are subclass of this.But different datasets have different attributes. So we cannot specifically instantiate them here
    # Subclasses can provide a list of default arguments which are automatically registered as attributes. This is a metaclass concept. I.e., some subclasses requires inheritence from more than one class. So we need to define it
    #to_do:- the concept of subclass registry is not yet cleared properly. study this.
    # TODO apparently there is a python 3.8 decorator that basically does this
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls
    #reading data path. The sequence dataset subclass is instantiated with inherited classes name, data_directory, and the properties of a sequence and keyword argument dataset_cfg
    def __init__(self, _name_, data_dir=None, tbptt=False, chunk_len=None, overlap_len=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Arguments for TBPTT: only used if tbptt is True and are passed to TBPTTDataLoader 
        # Not used right now
        self.tbptt = tbptt
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

        # Add all arguments to self
        init_args = self.init_defaults
        init_args.update(
            dataset_cfg
        )  # TODO this overrides the default dict which is bad
        for k, v in init_args.items():
            setattr(self, k, v)

        # train, val, test datasets must be set by class instantiation
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test"""
        raise NotImplementedError
    #function for splitting the datset to train, test and val. Check the dim of sequence x = (b,l,d):- (batch, length, dimension). We need to convert it to (x,y) pairs, where y is the label
    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  
        )

    @staticmethod
    def collate_fn(batch, resolution=1):
        """batch: list of (x, y) pairs"""
        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum(x.numel() for x in batch)
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution] # assume length is first axis after batch
                return x
            else:
                return torch.tensor(batch)

        x, y = zip(*batch)
        # Drop every nth sample
        # x = torch.stack(x, dim=0)[:, ::resolution]
        # y = torch.LongTensor(y)
        # y = torch.tensor(y)
        # y = torch.stack(y, dim=0)
        x = _collate(x, resolution=resolution)
        y = _collate(y, resolution=None)
        return x, y

    def train_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if train_resolution is None:
            train_resolution = [1]
        if not is_list(train_resolution):
            train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now"

        return self._dataloader(
            self.dataset_train,
            resolutions=train_resolution,
            shuffle=True,
            **kwargs,
        )[0]
    #here the _dataloader function is defined later(see below). Which is a torch.util.data.DataLoader object(This is what we needed before feeding into a network)
    #The below functions wraps all the data into the Dataloader
    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, train_resolution, eval_resolutions, **kwargs):
        if eval_resolutions is None:
            eval_resolutions = [1]
        if not is_list(eval_resolutions):
            eval_resolutions = [eval_resolutions]

        kwargs["shuffle"] = False if "shuffle" not in kwargs else kwargs["shuffle"]
        dataloaders = self._dataloader(
            dataset,
            resolutions=eval_resolutions,
            # shuffle=False,
            **kwargs,
        )

        return (
            {
                str(res) if res > 1 else None: dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None
            else None
        )
    #These None values are added and conditioned regularly throughout the code, inorder to avoid breakage while running the code, if no inputs are given. This is a good practice of error handling
    def _dataloader(self, dataset, resolutions, **loader_args):
        if dataset is None:
            return None

        DataLoader = torch.utils.data.DataLoader

        return [
            DataLoader(
                dataset=dataset,
                collate_fn=partial(self.collate_fn, resolution=resolution)
                if self.collate_fn is not None
                else None,
                **loader_args,
            )
            for resolution in resolutions
        ]

    def __str__(self):
        return self._name_


