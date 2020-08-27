from copy import deepcopy
from typing import Tuple

from PIL import Image
import numpy as np
import tensorflow as tf


def get_board_filters() -> tf.Tensor:
    """Creates a 4-D tensor of filters to identify edges within a chessboard.
    """
    # Create filters
    h_light_dark = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1],])
    h_dark_light = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1],])
    v_light_dark = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1],])
    v_dark_light = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1],])
    # Stack filters
    kernel = np.stack([h_light_dark, h_dark_light, v_light_dark, v_dark_light], axis=-1)
    # Convert to 4-D
    kernel = np.reshape(kernel, (3, 3, 1, 4))
    # Convert to tensor
    return tf.constant(kernel, dtype=tf.float32)


def preprocess_image(input_img: Image) -> tf.Tensor:
    img = deepcopy(input_img)
    # Convert to grayscale
    img = img.convert(mode="L")
    # Convert to array
    img = np.array(img)
    # Convert to 4-D
    img = np.reshape(img, (1, *img.shape, 1))
    # Convert to tensor
    return tf.constant(img, dtype=tf.float32)


def get_grid_indices(img: tf.Tensor) -> Tuple[np.array, np.array]:
    kernel = get_board_filters()

    # Apply convolution
    filtered_img = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding="SAME")
    # Clip values to 0-255 range
    filtered_img = tf.clip_by_value(filtered_img, 0, 255)
    # Collapse output channels into a single channel
    filtered_img = tf.math.reduce_max(filtered_img, axis=-1)
    # Remove first dimension
    filtered_img = tf.squeeze(filtered_img, axis=0)

    # Get indices of vertical and horizontal edges
    v_mean = tf.math.reduce_mean(filtered_img, axis=0)
    h_mean = tf.math.reduce_mean(filtered_img, axis=1)
    # Use 80% of maximum pixel value as threshold
    # NOTE: this threshold could be a problem if the "light" grid squares are too dark
    v_indices = tf.where(v_mean > (255 / 1.25))
    h_indices = tf.where(h_mean > (255 / 1.25))
    # Trim boundaries
    # NOTE: this may not work if image contains extra information outside board
    v_indices = v_indices[1:-1]
    h_indices = h_indices[1:-1]
    # Convert to 1-D numpy arrays
    v_indices = tf.squeeze(v_indices, axis=-1).numpy()
    h_indices = tf.squeeze(h_indices, axis=-1).numpy()

    # Verify number of grid lines
    if len(v_indices) != 14 or len(h_indices) != 14:
        raise ValueError("Detected an unexpected number of chessboard grid lines.")

    return v_indices, h_indices


def crop_board_image(
    board_img: tf.Tensor,
    grid_square_length: int,
    vert_indices: np.array,
    horiz_indices: np.array,
) -> tf.Tensor:
    # Crop image to grid squares
    v_start = vert_indices[0] - grid_square_length + 1
    v_end = vert_indices[-1] + grid_square_length
    h_start = horiz_indices[0] - grid_square_length + 1
    h_end = horiz_indices[-1] + grid_square_length
    return img[:, h_start:h_end, v_start:v_end, :]


def main():
    input_img = Image.open("img/board0.png")

    img = preprocess_image(input_img)

    # Identify gridline indices
    vert_indices, horiz_indices = get_grid_indices(img)

    # Determine length of grid square sides
    grid_square_length = vert_indices[2] - vert_indices[0]


if __name__ == "__main__":
    main()
