import torch
from .seqcode_dataset import SeqCodeDataset
from .seqcode_dataset import collate_fn as seqcode_collate
from .text_dataset import TextDataset
from .text_dataset import collate_fn as text_collate

from .classification_dataset import ClassificationDataset
from .classification_dataset import collate_fn as classification_collate
from .classification_dataset import non_seq_collate_fn as non_seq_classification_collate

from .seq_classification_dataset import SeqClassificationDataset
from .seq_classification_dataset import collate_fn as seq_classification_collate

from base import BaseDataLoader

import os


class SeqCodeDataLoader(BaseDataLoader):
    """
    Med2Vec Dataloader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        seed,
        **kwargs
    ):
        self.data_path = os.path.expanduser(data_dir)
        self.seed = seed
        self.batch_size = batch_size
        self.dataset = SeqCodeDataset(
            self.data_path, self.batch_size, **kwargs
        )
        super(SeqCodeDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=seqcode_collate,
            seed = self.seed
        )


class TextDataLoader(BaseDataLoader):
    """
    Mortality prediction task
    """

    def __init__(
        self,
        data_dir,
        text,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        training=True,
        **kwargs
    ):
        self.data_path = os.path.expanduser(data_dir)
        self.text = text
        self.batch_size = batch_size
        self.train = training
        self.dataset = TextDataset(
            self.data_path, self.text, self.batch_size, self.train, **kwargs
        )
        super(TextDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=text_collate,
        )


class ClassificationDataLoader(BaseDataLoader):
    """
    Length of stay & readmission prediction tasks
    """

    def __init__(
        self,
        text,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        y_label="los",
        training=True,
        vocab_fname="",
        balanced_data=False,
        sequential=True,
        **kwargs
    ):

        self.data_path = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.train = training
        self.text = text

        self.dataset = ClassificationDataset(
            self.data_path,
            self.text,
            self.batch_size,
            y_label,
            self.train,
            balanced_data=balanced_data,
            validation_split=validation_split,
            sequential=sequential,
            **kwargs
        )
        collate = classification_collate
        if not sequential:
            collate = non_seq_classification_collate

        super(ClassificationDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=collate,
        )

class SeqClassificationDataLoader(BaseDataLoader):
    """
    Length of stay, mortality and readmission prediction tasks using medical codes
    and/or demographics infomation only
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        y_label="mortality",
        seed=0,
        test=False,
        **kwargs
    ):

        self.data_path = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.test = test
        self.dataset = SeqClassificationDataset(
            self.data_path,
            self.batch_size,
            y_label,
            **kwargs
        )
        #self.pos_weight = torch.as_tensor(self.dataset.pos_weight).float()
        collate = seq_classification_collate
        
        super(SeqClassificationDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=collate,
            seed = self.seed,
            test = self.test
        )