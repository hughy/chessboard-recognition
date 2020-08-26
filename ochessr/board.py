from PIL import Image
import numpy as np
import tensorflow as tf


def main():
    img = Image.open("img/board0.png")
    # Convert to grayscale
    img = img.convert(mode="L")
    # Convert to array
    img = np.array(img)
    # Fix dimensions
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    # Convert to tensor
    img = tf.constant(img, dtype=tf.float32)

    # Create filters
    h_light_dark = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])[:, :, None, None]
    h_dark_light = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ])[:, :, None, None]
    v_light_dark = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])[:, :, None, None]
    v_dark_light = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ])[:, :, None, None]
    kernel_in = np.concatenate(
        [h_light_dark, h_dark_light, v_light_dark, v_dark_light],
        axis=-1
    )
    # Convert to tensor
    kernel = tf.constant(kernel_in, dtype=tf.float32)

    # Apply convolution
    filtered_img = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding="VALID")


if __name__ == "__main__":
    main()
