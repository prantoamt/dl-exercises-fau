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
        self.input_tensor = None
        self.output_tensor = None
        self.output_tensor_max_ind_list = []

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor

        stride_row, stride_col = self.stride_shape
        pooling_shape_row, pooling_shape_col = self.pooling_shape
        batch_size, channel, image_row, image_col = input_tensor.shape

        pool_row = (image_row-pooling_shape_row) // stride_row + 1
        pool_col = (image_col-pooling_shape_col) // stride_col + 1
        self.output_tensor = np.zeros((batch_size, channel, pool_row, pool_col))

        for batch_ind in range(batch_size):
            image_3d = input_tensor[batch_ind]

            for channel_ind in range(channel):
                image_2d = image_3d[channel_ind]

                max_pooling_2d_img = []
                qaurter_ind_row, qaurter_ind_col = 0, 0
                
                for start_row in range(0, image_row, stride_row):
                    end_row = start_row+pooling_shape_row
                    if end_row > image_row:
                        continue
                    
                    for start_col in range(0, image_col, stride_col):
                        end_col = start_col+pooling_shape_col
                        if end_col > image_col:
                            continue
                        
                        image_sliced = image_2d[start_row:end_row, start_col:end_col]
                        max_pool_value = np.amax(image_sliced)
                        max_pooling_2d_img.append(max_pool_value)

                        row_index, col_index = np.where(image_sliced == max_pool_value)
                        max_pool_index = (row_index[0]+qaurter_ind_row, col_index[0]+qaurter_ind_col)
                        self.output_tensor_max_ind_list.append(max_pool_index)
                        qaurter_ind_col += stride_col

                    qaurter_ind_col = 0
                    qaurter_ind_row += stride_row
                
                max_pooling_2d_img = np.array(max_pooling_2d_img).reshape(pool_row, pool_col)
                self.output_tensor[batch_ind, channel_ind, :, :] = max_pooling_2d_img
        
        return self.output_tensor


    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        output_error_tensor = np.zeros(self.input_tensor.shape)
        batch_size, channel, img_err_row, img_err_col = error_tensor.shape
        output_tensor_max_ind_counter = 0

        for batch_ind in range(batch_size):
            image_err_3d = error_tensor[batch_ind]

            for channel_ind in range(channel):
                image_err_2d = image_err_3d[channel_ind]

                for img_err_rows in image_err_2d:
                    for img_err_val in img_err_rows:
                        target_row, target_col = self.output_tensor_max_ind_list[output_tensor_max_ind_counter]

                        if output_error_tensor[batch_ind, channel_ind, target_row, target_col] == 0:
                            output_error_tensor[batch_ind, channel_ind, target_row, target_col] = img_err_val
                        else:
                            output_error_tensor[batch_ind, channel_ind, target_row, target_col] += img_err_val
                        output_tensor_max_ind_counter += 1
        
        return output_error_tensor