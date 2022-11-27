import os, random, math, json
from scipy.ndimage import rotate
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(
        self,
        file_path,
        label_path,
        batch_size,
        image_size,
        rotation=False,
        mirroring=False,
        shuffle=False,
    ):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

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

        # retrieve names of all images and shuffle if it's true
        self.all_images = [
            file
            for file in os.listdir(self.file_path)
            if os.path.isfile(os.path.join(self.file_path, file))
        ]
        if self.shuffle:
            random.shuffle(self.all_images)

        # load the labels into a dict
        self.labels_dict = None
        with open(self.label_path, "r") as f:
            self.labels_dict = json.load(f)

        self.dataset_pointer = (
            0  # for tracking the current position of the images array in each epoch
        )
        self.iteration_counter = 0  # for counting the iteration or number of epoch
        self.iteration_flag = False  # for checking whether one iteration is done or not

    def next(self):
        # if the iteration is done, then reset the dataset pointer, iteration flag
        # increase the iteration counter and shuffle again (if shuffle is true)
        if self.iteration_flag:
            self.dataset_pointer = 0
            self.iteration_counter += 1

            if self.shuffle:
                random.shuffle(self.all_images)

            self.iteration_flag = False

        images_list = None

        if self.dataset_pointer + self.batch_size <= len(self.all_images):
            # if the dataset pointer doesn't reach into the end of all image array then retrieve the batch
            images_list = self.all_images[
                self.dataset_pointer : self.dataset_pointer + self.batch_size
            ]

            # if dataset pointer is equal to the length of all image array, then change the iteration flag
            if self.dataset_pointer + self.batch_size == len(self.all_images):
                self.iteration_flag = True

            # update the dataset pointer
            self.dataset_pointer = self.dataset_pointer + self.batch_size
        else:
            img_arr1 = self.all_images[
                self.dataset_pointer :
            ]  # if the dataset pointer reach into the end of all image array then retrieve remaining images
            data_need = self.batch_size - (
                len(self.all_images) - self.dataset_pointer
            )  # calculate how much more data we need in last batch
            img_arr2 = self.all_images[
                :data_need
            ]  # retrieve the calculated data from the begining of the images array
            images_list = (
                img_arr1 + img_arr2
            )  # merge both of the array to create the last batch

            self.iteration_flag = True  # change the iteration flag

        # retrieve and load all images numpy files into that batch and their corresponding labels
        images, labels = list(), list()

        for img in images_list:
            labels.append(self.labels_dict[img.split(".")[0]])

            with open(os.path.join(self.file_path, img), "rb") as f:
                img = np.load(f)
                img = self.resize_image(
                    img
                )  # resize according to their desire dimensions

                if self.mirroring or self.rotation:
                    img = self.augment(
                        img
                    )  # augment data if the mirroring and rotation parameters are true

                images.append(img)

        return np.array(images), np.array(labels)

    def augment(self, img):
        # if mirroring parameter is true, then flip the image
        if self.mirroring:
            img = np.fliplr(img)

        # if rotation parameter is true, then rotate the image into a random degree betweeen 90, 180 and 270
        if self.rotation:
            rotation_deg = random.choice([90, 180, 270])
            img = rotate(img, angle=rotation_deg)

        return img

    def resize_image(self, img):
        # retrieve the desire height and width, then resize it according to the desire dimensions
        height, width, channel = self.image_size
        return resize(img, (height, width))

    def current_epoch(self):
        # returns the current epoch number
        return self.iteration_counter

    def class_name(self, x):
        # returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        num_images, num_cols = len(images), 3

        num_cols = min(num_images, num_cols)
        num_rows = math.ceil(num_images / num_cols)

        # height, width, _ = self.image_size
        # fig, axes        = plt.subplots(num_rows, num_cols, figsize=(height, width))
        fig, axes = plt.subplots(num_rows, num_cols)

        img_ind = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if img_ind >= num_images:
                    axes[row, col].set_visible(False)
                else:
                    axes[row, col].imshow(images[img_ind])
                    axes[row, col].set_title(self.class_name(labels[img_ind]))
                    axes[row, col].axis("off")
                img_ind += 1

        fig.tight_layout()
        plt.show()
