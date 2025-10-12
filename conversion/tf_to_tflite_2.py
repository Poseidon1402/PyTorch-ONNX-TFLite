from pathlib import Path

import tensorflow as tf

# Build the path relative to this script so the lookup works from anywhere.
saved_model_dir = (
    Path(__file__).resolve().parent.parent / "output" / "mobilenet_ssd"
)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir.as_posix())
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
