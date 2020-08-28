import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import preprocessing


DATASET_DIRECTORY = "data/pieces"
MODEL_FILEPATH = "model/piece_model.pb"
SHUFFLE_SEED = 123
TRAINING_EPOCHS = 16
VALIDATION_SPLIT = 0.2


CLASS_LABELS = [
    "_",
    "bB",
    "bK",
    "bN",
    "bP",
    "bQ",
    "bR",
    "wB",
    "wK",
    "wN",
    "wP",
    "wQ",
    "wR",
]


def get_piece_model() -> tf.keras.Model:
    """Constructs a model for classifying chess piece images.

    The model architecture is based on the LeNet-5 model for classifying
    hand-written digits.
    """
    return models.Sequential(
        [
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255, input_shape=(32, 32, 1)
            ),
            layers.Conv2D(32, (3, 3), activation="tanh"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="tanh"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="tanh"),
            layers.Flatten(),
            layers.Dense(64, activation="tanh"),
            layers.Dense(len(CLASS_LABELS)),
        ]
    )


def get_dataset(subset: str) -> tf.data.Dataset:
    """Loads the dataset of chess piece images.

    The value of `subset` determines which of the training or validation
    dataset to load.
    """
    if subset not in {"training", "validation"}:
        raise ValueError(
            f"Invalid dataset subset '{subset}'. Use 'training' or 'validation'."
        )
    return preprocessing.image_dataset_from_directory(
        DATASET_DIRECTORY,
        validation_split=VALIDATION_SPLIT,
        subset=subset,
        seed=SHUFFLE_SEED,
        image_size=(32, 32),
        batch_size=64,
        color_mode="grayscale",
    )


def train(model: tf.keras.Model, save_model: bool = True) -> tf.keras.Model:
    """Trains the classification model on the chess piece image dataset.
    """
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    train_dataset = get_dataset("training")
    validation_dataset = get_dataset("validation")

    model.fit(train_dataset, validation_data=validation_dataset, epochs=TRAINING_EPOCHS)

    if save_model:
        model.save(MODEL_FILEPATH)

    return model


def main():
    model = get_piece_model()
    model = train(model)


if __name__ == "__main__":
    main()
