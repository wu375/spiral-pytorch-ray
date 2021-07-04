from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class RealImageDataset(Dataset):
    def __init__(self, dataset, batch_size=64, size=64):
        super(Dataset, self).__init__()
        self.size = size
        self.batch_size = batch_size
        if dataset == 'mnist':
            # self.image_paths = list(Path('../datasets/mnist_png/').resolve().glob('*/*/*.png'))
            self.image_paths = list(Path('../datasets/mnist_png/').resolve().glob('*/4/*.png'))
        elif dataset == 'celeba':
            self.image_paths = list(Path('../datasets/celeba_hq/').resolve().glob('*/*/*.jpg'))
        else:
            assert False, 'unknown dataset'

        self.dataset = dataset

    def __len__(self):
        return self.batch_size * 20

    def __getitem__(self, idx):
        image_id = np.random.randint(low=0, high=len(self.image_paths))
        # image_id = 46
        image = self.image_paths[image_id]
        image = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if self.dataset == 'celeba':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        if self.dataset == 'mnist':
            return 1 - (image / 255.0)
        return image / 255.0

def inf_loader_wrapper(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def get_real_image_loader(dataset, batch_size=64):
    loader = DataLoader(
        RealImageDataset(dataset=dataset, batch_size=batch_size),
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
    )
    loader = inf_loader_wrapper(loader)
    return loader

if __name__ == '__main__':
    # seed_loader = DataLoader(
    #     SeedDataset(pad_target=pad_target, seed_shape=(72,72,16)),
    #     batch_size=8,
    #     num_workers=4,
    #     shuffle=True,
    #     worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
    # )
    # seed_loader = inf_loader_wrapper(seed_loader)
    pass