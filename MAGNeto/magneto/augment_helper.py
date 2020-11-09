import random

import numpy as np
import pandas as pd
from scipy.special import softmax
from torchvision import transforms

from magneto.autoaugment import ImageNetPolicy


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SHAPE = 112

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SHAPE, scale=(0.3, 1.0)),
    transforms.RandomHorizontalFlip(),
    ImageNetPolicy(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

val_transform = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.CenterCrop(INPUT_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


class TagAugmentation():
    def __init__(
        self,
        vocab_path: str,
        drop: float = 0.0,
        add: float = 0.0,
        path: str = None
    ):
        self.vocab = pd.read_csv(
            vocab_path, keep_default_na=False, na_values=['']).word.tolist()
        self.drop, self.add = drop, add

    def __call__(self, tags: np.array, label: np.array) -> (np.array, np.array):
        '''
        input:
            + tags: raw tags.
            + label: raw label.
        output:
            Processed tags and corresponding label.
        '''
        self.tags, self.label = tags, label
        self._seperate_indices()

        # NOTE: Dropping must be performed prior to adding process
        if self.drop:
            self.tags, self.label = self._drop_tag()

        if self.add:
            self.tags, self.label = self._add_tag()

        return self.tags, self.label

    def _get_num(self, prob: float):
        return random.randint(0, min(int(prob * len(self.unimportant_indices)), len(self.vocab)))

    def _seperate_indices(self):
        unimportant_mask = self.label == 0
        self.unimportant_indices = np.array(range(len(self.tags)))[
            unimportant_mask]
        self.important_indices = np.array(range(len(self.tags)))[
            np.logical_not(unimportant_mask)]

    def _drop_tag(self):
        # Randomly select the number of unimportant tags to keep
        num_unimportant_drop = self._get_num(self.drop)
        num_unimportant_keep = len(
            self.unimportant_indices) - num_unimportant_drop

        # Randomly choose indices of unimportant tags to keep based on the number above
        unimportant_keep_indices = np.array(random.sample(
            list(self.unimportant_indices), k=num_unimportant_keep), dtype=int)
        keep_indices = np.concatenate(
            (unimportant_keep_indices, self.important_indices))
        keep_indices.sort()

        return self.tags[keep_indices], self.label[keep_indices]

    def _add_tag(self):
        num_add = self._get_num(self.add)

        tags_add = []
        sampled_tags = 0
        while sampled_tags < num_add:
            noise_tags = random.sample(self.vocab, k=num_add - sampled_tags)
            valid_tags = [t for t in noise_tags if t not in self.tags]
            tags_add += valid_tags
            sampled_tags = len(tags_add)

        tags = np.concatenate((self.tags, np.asarray(tags_add)))
        label = np.concatenate((self.label, np.zeros(num_add)))

        return tags, label
