from typing import Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mode: str,
        transform: tv.transforms.Compose = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.mode = mode
        self._transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = str(Path("ex_4/src_to_implement/", self.data.iloc[index, 0]))
        image_label = torch.as_tensor(
            [self.data.iloc[index, 1], self.data.iloc[index, 2]]
        )
        image = imread(image_path)
        image = gray2rgb(image=image)
        train_transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=torch.tensor(train_mean),
                    std=torch.tensor(train_std),
                    inplace=True,
                ),
            ]
        )
        image = train_transform(image)
        if self._transform:
            image = self._transform(image)
        return image, image_label


def __show_data(
    data_loader: torch.utils.data.DataLoader, num_of_data: Optional[int] = 6
):
    for images, image_names in data_loader:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(tv.utils.make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break
    plt.show()


csv_data = pd.read_csv("ex_4/src_to_implement/data.csv", sep=";", skiprows=1)
cd = ChallengeDataset(csv_data, "train")
val_data = torch.utils.data.DataLoader(cd, batch_size=100)

__show_data(val_data)
