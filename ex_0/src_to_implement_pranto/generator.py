# Python imports
import os
import json
from random import randrange, choice
from typing import List, Optional

# self imports

# Other imports
from skimage.transform import resize, rotate
import numpy as np
import matplotlib.pyplot as plt


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(
        self,
        file_path: str,
        label_path: str,
        batch_size: int,
        image_size: List,
        rotation: Optional[bool] = False,
        mirroring: Optional[bool] = False,
        shuffle: Optional[bool] = False,
    ):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self._init_setup()

    def _init_setup(self) -> None:
        """
        Does the initial setups
        """
        self.all_images = os.listdir(self.file_path)
        self._epoch_no = 0
        self.start_index = 0
        self.end_index_plus_one = self.start_index + self.batch_size
        self.rotation_options = [90.0, 180.0, 270.0]
        self.load_json()
        if self.shuffle:
            self._shuffle_slice()

    def load_json(self):
        with open(self.label_path) as f:
            self.all_labels = json.load(f)

    def next(self) -> tuple:
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # self._increment_epoch()

        if self.start_index > len(self.all_images):
            self._increment_epoch()

        image_names = []
        images = []
        labels = []
        image_names = self._get_image_names()

        for image in image_names:
            image_array = np.load(os.path.join(self.file_path, image))
            image_array = resize(image=image_array, output_shape=self.image_size)
            if self.rotation:
                image_array = self._rotate_image(image_as_np_array=image_array)
            if self.mirroring:
                image_array = self._mirror_image(image_as_np_array=image_array)
            images.append(image_array)
            image_number = image.split(".")[0]
            labels.append(self.all_labels.get(image_number))

        return np.array(images, dtype="float64"), labels

    def _rotate_image(self, image_as_np_array: np.ndarray) -> np.ndarray:
        rotation_angle = choice(self.rotation_options)
        rotated_image = rotate(image_as_np_array, rotation_angle)
        return rotated_image

    def _mirror_image(self, image_as_np_array: np.ndarray) -> np.ndarray:
        return np.fliplr(image_as_np_array)

    def _get_image_names(self) -> List:
        """
        Gets the image names to be loaded. If Batch size is bigger that the
        data, adjust the start and end index and asks the _manage_indices() to
        reset the indices and take data from index 0 to fill up the remaining
        data.
        Params: None
        Return: image_names: List
        """
        image_names = []
        remainig_batch_members = 0
        numbers_of_remainig_images = self._get_number_of_remaing_img()
        if self.batch_size > numbers_of_remainig_images:
            remainig_batch_members = self.batch_size - numbers_of_remainig_images
            self._manage_image_name_indices(temp_batch_size=numbers_of_remainig_images)

        if self.shuffle:
            self._shuffle_slice()

        image_names = (
            image_names + self.all_images[self.start_index : self.end_index_plus_one]
        )

        if remainig_batch_members:
            self._manage_image_name_indices(
                remainig_batch_members=remainig_batch_members
            )
            image_names = (
                image_names
                + self.all_images[self.start_index : self.end_index_plus_one]
            )

        self._manage_image_name_indices()

        return image_names

    def _get_number_of_remaing_img(self) -> int:
        """
        Returns the number of remaining images from last accessed index
        Params: None
        Return: number_of_remaining_img: int
        """
        return len(self.all_images[self.start_index :])

    def _manage_image_name_indices(
        self,
        temp_batch_size: Optional[int] = 0,
        remainig_batch_members: Optional[int] = 0,
    ) -> None:
        """
        Manage indices (last accessed index and batch size) for each batch.
        If there is any remaining batch members that could not be included due
        to lacking of data, it resets the index to take data from begining.
        Params: remaining_batch_members [Optional] : int
        Return: None
        """
        prev_start_index = self.start_index
        self.start_index = prev_start_index + self.end_index_plus_one
        self.end_index_plus_one = self.start_index + self.batch_size

        if remainig_batch_members > 0:
            self._increment_epoch()
            self.end_index_plus_one = self.start_index + remainig_batch_members

        if temp_batch_size > 0:
            self.start_index = prev_start_index
            self.end_index_plus_one = self.start_index + temp_batch_size

        return None

    def _increment_epoch(self):
        self.start_index = 0
        self.end_index_plus_one = self.start_index + self.batch_size
        self._epoch_no += 1

    def _shuffle_slice(self) -> None:
        """
        Shuffles the all_images list from start_index to batch size
        Params: None
        Return: None
        """
        i = self.start_index
        numbers_of_remainig_images = self._get_number_of_remaing_img()
        while i < numbers_of_remainig_images - 1:
            idx = randrange(i, numbers_of_remainig_images)
            self.all_images[i], self.all_images[idx] = (
                self.all_images[idx],
                self.all_images[i],
            )
            i += 1
        return None

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number
        return self._epoch_no

    def class_name(self, x: int) -> str:
        # This function returns the class name for a specific input
        return self.class_dict.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method

        images, labels = self.next()
        fig, axes = plt.subplots(3, 4, figsize=(10, 10))
        if isinstance(axes, np.ndarray):
            list_axes = list(axes.flat)
        else:
            list_axes = [axes]

        for i in range(self.batch_size):
            list_axes[i].imshow(images[i])
            list_axes[i].set_title(self.class_name(labels[i]))
            list_axes[i].axes.get_xaxis().set_visible(False)
            list_axes[i].axes.get_yaxis().set_visible(False)
        plt.show()
