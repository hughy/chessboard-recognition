from PIL import Image
import tensorflow as tf

from ochessr import board
from ochessr import piece_model


class ChessBoardLayer(tf.keras.layers.Layer):
    def call(self, input):
        return board.crop_board_image(input)


class ChessPieceLayer(tf.keras.layers.Layer):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model

    def call(self, input):
        return predict_board_labels(self.model, input)


class FENLayer(tf.keras.layers.Layer):
    def call(self, input):
        return board_to_fen(input)


def predict_board_labels(model: tf.keras.Model, img: tf.Tensor) -> tf.Tensor:
    # Resize image for piece_model input
    img = tf.image.resize(img, (32 * 8, 32 * 8))
    # Extract image patches
    patch_shape = (1, 32, 32, 1)
    patches = tf.image.extract_patches(
        img,
        sizes=patch_shape,
        strides=patch_shape,
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    # Reshape patches for input into piece_model
    board_squares = tf.reshape(patches, (64, 32, 32, 1))
    predictions = model.predict(board_squares)
    # Identify predicted chess piece
    classes = tf.math.argmax(predictions, axis=-1)
    labels = tf.map_fn(
        piece_model.CLASS_LABEL_LIST.__getitem__, classes, dtype=tf.string
    )
    return tf.reshape(labels, (8, 8))


def board_to_fen(board_labels: tf.Tensor) -> str:
    board_labels_arr = board_labels.numpy()
    row_strings = []
    for row in board_labels_arr:
        row_string = ""
        empty_squares = 0
        for label in row:
            # TensorFlow uses byte strings so convert to unicode
            unicode_label = str(label, "utf-8")
            if unicode_label == "_":
                empty_squares += 1
            else:
                row_string += str(empty_squares) if empty_squares else ""
                row_string += unicode_label
                empty_squares = 0

        row_string += str(empty_squares) if empty_squares else ""
        row_strings.append(row_string)

    return "/".join(row_strings)


def ochessr_model(input_path):
    model = tf.keras.models.Sequential([
        ChessBoardLayer(),
        ChessPieceLayer(piece_model.load_model()),
        FENLayer(),
    ])
    img = board.preprocess_image(Image.open(input_path))
    fen = model(img)
    print(fen)


def main(input_path: str):
    # Load image
    img = Image.open(input_path)
    # Preprocess image/crop to board
    img = board.preprocess_image(img)
    img = board.crop_board_image(img)
    # Load piece_model
    model = piece_model.load_model()
    # Predict chess pieces
    board_labels = predict_board_labels(model, img)
    # Convert to FEN
    fen = board_to_fen(board_labels)
    print(fen)


if __name__ == "__main__":
    main("img/board0.png")
