from PIL import Image
import numpy as np
import tensorflow as tf


def get_board_filters() -> tf.Tensor:
    """Creates a 4-D tensor of filters to identify edges within a chessboard.
    """
    # Create filters
    h_light_dark = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])
    h_dark_light = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ])
    v_light_dark = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])
    v_dark_light = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ])
    # Stack filters
    kernel = np.stack(
        [h_light_dark, h_dark_light, v_light_dark, v_dark_light],
        axis=-1
    )
    # Convert to 4-D
    kernel = np.reshape(kernel, (3, 3, 1, 4))
    # Convert to tensor
    return tf.constant(kernel, dtype=tf.float32)


def preprocess_image(img: Image) -> tf.Tensor:
    # Convert to grayscale
    img = img.convert(mode="L")
    # Convert to array
    img = np.array(img)
    # Convert to 4-D
    img = np.reshape(img, (1, *img.shape, 1))
    # Convert to tensor
    return tf.constant(img, dtype=tf.float32)

def main():
    img = Image.open("img/board0.png")
    img = preprocess_image(img)
    kernel = get_board_filters()

    # Apply convolution
    filtered_img = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding="SAME")
    # Clip values to 0-255 range
    filtered_img = tf.clip_by_value(filtered_img, 0, 255)
    # Collapse output channels into a single channel
    filtered_img = tf.math.reduce_max(filtered_img, axis=-1)

    # Get indices of vertical and horizontal edges
    v_mean = tf.math.reduce_mean(filtered_img, axis=1)
    h_mean = tf.math.reduce_mean(filtered_img, axis=2)
    # Use 3/4 of maximum pixel value as threshold
    # NOTE: this threshold could be a problem if the "light" grid squares are too dark
    v_indices = tf.where(v_mean > (255 / 1.5))
    h_indices = tf.where(h_mean > (255 / 1.5))


if __name__ == "__main__":
    main()
