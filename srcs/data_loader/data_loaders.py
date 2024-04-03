import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from torchvision import transforms as T 

class SignDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # self.transform = T.compose(
        #     T.Blur(blur_limit=5, p=0.2)
        #     T.Normalize(mean=(0.0118), std=(0.0036)),
        #     T.ToTensorV2(),
        # ) # если есть аугментации

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].to_numpy().reshape(28, 28)
        label = self.data.iloc[idx]['label']

        image = np.stack([image, image, image])
        # image.transform()
        return torch.tensor(image).float(), torch.tensor(label)


def get_sign_dataloader(
        csv_path_train, csv_path_val, batch_size, shuffle=True, num_workers=1,
    ):
    train_dataset = SignDataset(csv_file=csv_path_train)
    val_dataset = SignDataset(csv_file=csv_path_val)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        csv_path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(csv_file=csv_path_test)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
