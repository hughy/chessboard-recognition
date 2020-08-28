from copy import deepcopy
from typing import List
from typing import Optional
from typing import Tuple

from PIL import Image
import numpy as np
import tensorflow as tf


# Chessboard squares must be at least 16 pixels in length
MIN_SQUARE_LENGTH = 16


def preprocess_image(input_img: Image) -> tf.Tensor:
    """Preprocess the input image.
    """
    img = deepcopy(input_img)
    # Convert to grayscale
    img = img.convert(mode="L")
    # Convert to array
    img = np.array(img)
    # Convert to 4-D
    img = np.reshape(img, (1, *img.shape, 1))
    # Convert to tensor
    return tf.constant(img, dtype=tf.float32)


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


def detect_grid_indices(img: tf.Tensor) -> Tuple[np.array, np.array]:
    """Detects image indices of chessboard gridlines.
    """
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
    v_indices = tf.where(v_mean > (255 / 1.25))
    h_indices = tf.where(h_mean > (255 / 1.25))
    # Convert to 1-D numpy arrays
    v_indices = tf.squeeze(v_indices, axis=-1).numpy()
    h_indices = tf.squeeze(h_indices, axis=-1).numpy()

    return filter_grid_indices(v_indices), filter_grid_indices(h_indices)


def filter_grid_indices(line_indices: np.array) -> np.array:
    possible_indices = []
    for i, index in enumerate(line_indices[1:-1], start=1):
        # Convolution filters create two adjacent lines demarcating grid edges
        if (
            index - line_indices[i - 1] > 1
            or line_indices[i + 1] - index < MIN_SQUARE_LENGTH
        ):
            continue
        possible_indices.append(index)

    grid_indices = filter_evenly_spaced_indices(possible_indices)

    if grid_indices is None:
        raise ValueError("Failed to detect grid lines in chessboard!")

    return np.array(grid_indices)


def filter_evenly_spaced_indices(possible_indices: List[int]) -> Optional[List[int]]:
    """Finds a list of seven evenly-spaced indices from the input list, if any.
    """
    set_possible_indices = set(possible_indices)
    for start in possible_indices[:-6]:
        for end in list(reversed(possible_indices))[:-6]:
            space_length = (end - start) // 6
            # Must be space between edge of image and first index
            if start < space_length:
                continue
            grid_indices = list(range(start, end + 1, space_length))
            set_grid_indices = set(grid_indices)
            if set_grid_indices.issubset(set_possible_indices):
                return grid_indices

    return None


def get_cropped_board_image(input_path: str) -> tf.Tensor:
    """Detect chessboard gridlines in the input image and crop to the board edges.
    """
    input_img = Image.open(input_path)
    img = preprocess_image(input_img)
    # Identify gridline indices
    vert_indices, horiz_indices = detect_grid_indices(img)
    # Determine length of grid square sides
    square_length = vert_indices[1] - vert_indices[0]
    # Crop image to grid squares
    v_start = vert_indices[0] - square_length
    v_end = vert_indices[-1] + square_length
    h_start = horiz_indices[0] - square_length
    h_end = horiz_indices[-1] + square_length
    return img[:, h_start:h_end, v_start:v_end, :]


if __name__ == "__main__":
    cropped_image = get_cropped_board_image("img/board0.png")
