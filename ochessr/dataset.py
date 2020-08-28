import os
import uuid
from typing import List

from PIL import Image
import tensorflow as tf

from ochessr.board import get_cropped_board_image


BOARD_PATH_FMT = "data/boards/board{i}.png"
SCREENSHOT_PATH_FMT = "data/screenshots/board{i}.png"
SQUARE_DIR_FMT = "data/pieces/{label}/"
SQUARE_PATH_FMT = "data/pieces/{label}/{id}.png"


# Mapping from board indices to class labels in screenshots
PIECE_LABEL_MAPPING = [
    ["bK", "bQ", "bR", "bB", "bN", "bP", "bK", "bQ"],
    ["bR", "bB", "bN", "bP", "bK", "bQ", "bR", "bB"],
    ["bN", "bP", "bK", "bQ", "bR", "bB", "bN", "bP"],
    ["bK", "bQ", "bR", "bB", "bN", "bP", "_", "_"],
    ["wK", "wQ", "wR", "wB", "wN", "wP", "_", "_"],
    ["wN", "wP", "wK", "wQ", "wR", "wB", "wN", "wP"],
    ["wR", "wB", "wN", "wP", "wK", "wQ", "wR", "wB"],
    ["wK", "wQ", "wR", "wB", "wN", "wP", "wK", "wQ"],
]


def save_tensor_as_image(tensor: tf.Tensor, path: str):
    """Saves a tensor to the given path as a grayscale image.
    """
    image = Image.fromarray(tensor.numpy())
    image.convert(mode="L").save(path)


def generate_image_dataset(screenshot_indices: List[int]):
    """Generates cropped chessboard images and images of chessboard squares.
    """
    for i in screenshot_indices:
        # Load screenshot and crop to board
        try:
            board = get_cropped_board_image(SCREENSHOT_PATH_FMT.format(i=i))
        except ValueError:
            print(f"Failed to crop screenshot{i}.png")
            continue
        # Save board image
        save_tensor_as_image(board[0, :, :, 0], BOARD_PATH_FMT.format(i=i))
        # Extract patches
        square_length = int(board.shape[1] / 8)
        patch_shape = (1, square_length, square_length, 1)
        patches = tf.image.extract_patches(
            board,
            sizes=patch_shape,
            strides=patch_shape,
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        for row in range(8):
            for col in range(8):
                label = PIECE_LABEL_MAPPING[row][col]
                square = patches[0, row, col, :]
                square = tf.reshape(square, (square_length, square_length))
                # Save board square image
                square_image_id = "_".join([str(i), label, uuid.uuid4().hex])
                save_tensor_as_image(
                    square, SQUARE_PATH_FMT.format(label=label, id=square_image_id)
                )


def main():
    # Create all output directories
    board_dir = "data/boards"
    os.makedirs(board_dir, exist_ok=True)
    for row in PIECE_LABEL_MAPPING:
        for label in row:
            square_dir = SQUARE_DIR_FMT.format(label=label)
            os.makedirs(square_dir, exist_ok=True)

    # Upsample the default piece set by including that screenshot multiple times
    screenshot_indices = [0, 0, 0, 0] + list(range(22))
    generate_image_dataset(screenshot_indices=screenshot_indices)


if __name__ == "__main__":
    main()
