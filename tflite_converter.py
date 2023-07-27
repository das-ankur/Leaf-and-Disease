import tensorflow as tf


def tflite_converter(saved_model_path: str, save_path: str):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    # Save the model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    tflite_converter(
        saved_model_path="saved_model",
        save_path="tflite_model/model.tflite"
    )
