import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from magneto.augment_helper import train_transform, val_transform, TagAugmentation


class TagAndImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vocab_path: str,
        img_dir: str,
        max_len: int,
        has_label: bool = True,
        return_item_id: bool = False,
        tag_preprocess_fn: object = None,
        img_preprocess_fn: object = None,
    ):
        '''
        input:
            + csv_path: path to the csv file that contains "image_id", "tags"[, "label"].
            + vocab_path: path to the csv file that contains the vocabulary of the dataset.
            + img_dir: the directory that contains corresponding images.
            + max_len: the maximum number of tags.
            + has_label: whether to prepare and return label or not.
            + return_item_id: whether to return item_id or not.
            + tag_preprocess_fn: the preprocessing func for tags; only support when having label.
            + img_preprocess_fn: the preprocessing func for image.
        '''
        df = pd.read_csv(csv_path)

        self.has_label = has_label
        self.return_item_id = return_item_id

        self.list_of_tags = df['tags'].apply(
            lambda x: np.array(x.split(','))).tolist()
        self.list_of_image_path = df['item_id'].map(
            lambda x: os.path.join(img_dir, str(x) + '.jpg')).tolist()
        if self.has_label:
            self.list_of_label = df['label'].apply(
                lambda x: np.array(x.split(','), dtype=np.float32)).tolist()
        if self.return_item_id:
            self.list_of_item_id = df['item_id'].tolist()

        self.vocab = pd.read_csv(
            vocab_path, keep_default_na=False, na_values=[''])
        self.word_to_index = self.vocab.set_index('word')
        self.vocab_size = len(self.vocab)

        self.max_num_of_tags = max_len
        self.tag_preprocess_fn = tag_preprocess_fn
        self.img_preprocess_fn = img_preprocess_fn

    def __len__(self) -> int:
        return len(self.list_of_tags)

    def __getitem__(self, idx: object) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        '''
        input:
            + idx: item's index.
        output:
            + image: self explanatory.
            + vectors: embedding vectors of tags.
            + label: corresponding label (only returned when being provided).
            + mask: generated mask used to mask-out padding positions.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image
        image_path = self.list_of_image_path[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return self.__getitem__(random.randrange(self.__len__()))

        if self.img_preprocess_fn is not None:
            image = self.img_preprocess_fn(image)

        # Get indices of tags, corresponding mask and label (if provided)
        tags = self.list_of_tags[idx]
        if self.has_label:
            label = self.list_of_label[idx]

            assert len(tags) == len(label)

            if self.tag_preprocess_fn is not None:
                tags, label = self.tag_preprocess_fn(tags, label)
        if self.return_item_id:
            item_id = self.list_of_item_id[idx]

        # Create default mask
        mask = torch.zeros(self.max_num_of_tags, dtype=torch.bool)

        # Fixed the number of tags
        if len(tags) >= self.max_num_of_tags:
            # Get top N
            tags = tags[:self.max_num_of_tags]
            indices = self.word_to_index.loc[tags, 'index']
            indices = torch.tensor(indices, dtype=torch.int64)
            if self.has_label:
                label = torch.tensor(
                    label[:self.max_num_of_tags],
                    dtype=torch.float32
                )
        else:
            indices = self.word_to_index.loc[tags, 'index']

            # Right-padding
            # Padding idx will be n where n = vocab_size
            padding_vector = np.ones(
                self.max_num_of_tags, dtype=np.int64) * (self.vocab_size)
            padding_vector[:len(tags)] = indices
            indices = torch.tensor(padding_vector, dtype=torch.int64)

            mask[len(tags):] = True

            if self.has_label:
                zeros_vector = np.zeros(
                    self.max_num_of_tags, dtype=np.float32)
                zeros_vector[:len(tags)] += label
                label = torch.tensor(zeros_vector, dtype=torch.float32)

        results = [image, indices, mask]
        if self.has_label:
            results.append(label)
        if self.return_item_id:
            results.append(item_id)

        return results


def get_dataloaders(
    train_csv_path: str,
    val_csv_path: str,
    vocab_path: str,
    img_dir: str,
    tagaug_add_max_ratio: float,
    tagaug_drop_max_ratio: float,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    max_len: int = 100,
    num_workers: int = 0,
    pin_memory: bool = True
) -> (DataLoader, DataLoader):
    '''
    input:
        + train_csv_path: path to the csv file of the training dataset.
        + val_csv_path: path to the csv file of the validation dataset.
        + vocab_path: path to the csv file that contains the vocabulary of the dataset.
        + img_dir: the directory that contains all images for training and validation sets.
        + tagaug_add_max_ratio: the maximum ratio between the number of adding tags and non-important ones.
        + tagaug_drop_max_ratio: the maximum ratio between the number of dropping tags and non-important ones.
        + train_batch_size: the batch-size of the training dataloader.
        + val_batch_size: the batch-size of the validation dataloader.
        + max_len: the maximum length for each set of tags.
        + num_workers: the number of workers used to load data.
        + pin_memory: the pin_memory param of PyTorch's DataLoader class.
    output:
        the dataloaders for the training and validation sets.
    '''
    train_dataset = TagAndImageDataset(
        csv_path=train_csv_path,
        vocab_path=vocab_path,
        img_dir=img_dir,
        max_len=max_len,
        tag_preprocess_fn=TagAugmentation(
            vocab_path=vocab_path,
            add=tagaug_add_max_ratio,
            drop=tagaug_drop_max_ratio
        ) if (tagaug_add_max_ratio or tagaug_drop_max_ratio) else None,  # Only use when necessary
        img_preprocess_fn=train_transform
    )
    val_dataset = TagAndImageDataset(
        csv_path=val_csv_path,
        vocab_path=vocab_path,
        img_dir=img_dir,
        max_len=max_len,
        img_preprocess_fn=val_transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, val_dataloader, train_dataset.vocab_size
