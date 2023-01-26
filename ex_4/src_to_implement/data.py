from typing import Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
import matplotlib.pyplot as plt

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
        image = imread(image_path)
        image = gray2rgb(image=image)
        train_transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    torch.tensor(train_mean), torch.tensor(train_std), inplace=True
                ),
            ]
        )
        image = train_transform(image)
        if self._transform:
            image = self._transform(image)
        image_name = self.data.iloc[index, 0].split("/")[1]
        return image, image_name


def __show_data(
    data_loader: torch.utils.data.DataLoader, num_of_data: Optional[int] = 6
):
    fig = plt.figure()
    position = 1
    iteration = 1
    for img, img_name in data_loader:
        img = torch.detach(img[0]).numpy()
        img = img.transpose(1, 2, 0)
        ax = plt.subplot(2, 3, position)
        plt.tight_layout()
        ax.set_title(f"Sample {img_name}")
        ax.axis("off")
        plt.imshow((img * 255).astype("uint8"))
        if position == 3:
            position = 1
        else:
            position += 1
        if iteration == num_of_data:
            break
        iteration += 1
    plt.show()


csv_data = pd.read_csv("ex_4/src_to_implement/data.csv", sep=";", skiprows=1)
cd = ChallengeDataset(csv_data, "train")

val_data = torch.utils.data.DataLoader(cd, batch_size=1)

# __show_data(val_data)
