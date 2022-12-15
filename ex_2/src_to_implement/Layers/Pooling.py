# Python imports
from typing import Union

# Third party imports
import numpy as np

# Self imports
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape: Union[tuple, int], pooling_shape: int) -> None:
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        print(f"Input Tensor Shape: {input_tensor.shape}")
        print(f"Stride Shape: {self.stride_shape}")
        print(f"Pooling Shape: {self.pooling_shape}")
        stride_horiz, stride_vert = self.stride_shape
        pooling_shape_horiz, pooling_shape_vert = self.pooling_shape
        batch_size, channel, image_row, image_col = input_tensor.shape

        print("*"*100)

        output_tensor = np.array([batch_size, channel])
        for batch_ind in range(batch_size):
            image_3d = input_tensor[batch_ind]
            print(f"3D Image Shape: {image_3d.shape}")

            max_pooling_3d_img = np.array([])
            for channel_ind in range(channel):
                image_2d = image_3d[channel_ind]
                print(f"2D Image Shape: {image_2d.shape}")
                print(image_2d)
                max_pooling_2d_img = []

                for start_row in range(0, image_row, stride_vert):
                    end_row = start_row+pooling_shape_vert
                    if end_row > image_row:
                        continue
                    
                    for start_col in range(0, image_col, stride_horiz):
                        end_col = start_col+pooling_shape_horiz
                        if end_col > image_col:
                            continue
                        # print(image_2d[start_row:end_row, start_col:end_col], np.amax(image_2d[start_row:end_row, start_col:end_col]))
                        max_pooling_2d_img.append(np.amax(image_2d[start_row:end_row, start_col:end_col]))
                
                pool_row = (image_row-pooling_shape_vert) // stride_vert + 1
                pool_col = (image_col-pooling_shape_horiz) // stride_horiz + 1
                max_pooling_2d_img = np.array(max_pooling_2d_img).reshape(pool_row, pool_col)
                print(max_pooling_2d_img)
                max_pooling_3d_img = np.append(max_pooling_3d_img, max_pooling_2d_img)
                print(max_pooling_3d_img.shape)
                # start_row, start_col = 0, 0
                # end_row, end_col = start_row+pooling_shape_vert, start_col+pooling_shape_horiz
                # print(image_2d[start_row:end_row, start_col:end_col])

                # start_col = start_col+stride_horiz
                # end_col = start_col+pooling_shape_horiz
                # print(image_2d[start_row:end_row, start_col:end_col])

                # start_row, start_col = start_row+stride_vert, 0
                # end_row, end_col = start_row+pooling_shape_vert, start_col+pooling_shape_horiz
                # print(image_2d[start_row:end_row, start_col:end_col])

                # start_col = start_col+stride_horiz
                # end_col = start_col+pooling_shape_horiz
                # print(image_2d[start_row:end_row, start_col:end_col])
        print(output_tensor.shape)
        print("*"*100)


    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        pass
