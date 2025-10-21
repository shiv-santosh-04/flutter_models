import tensorflow as tf
from keras_ocr.detection import Detector
from keras_ocr.recognition import Recognizer

print("--- [1/4] Re-creating model architectures... ---")
detector = Detector()
recognizer = Recognizer()

print("--- [2/4] Loading saved weights into models... ---")
detector.model.load_weights('detector_weights.h5')
recognizer.model.load_weights('recognizer_weights.h5')

# --- Convert the Detector Model (This is fine) ---
print("--- [3/4] Converting detector model to TFLite... ---")
converter_detector = tf.lite.TFLiteConverter.from_keras_model(detector.model)
tflite_detector_model = converter_detector.convert()
with open('ocr_detector.tflite', 'wb') as f:
    f.write(tflite_detector_model)
print("      -> Saved as ocr_detector.tflite")


# --- Convert the Recognizer Model (The Only Working Method) ---
print("--- [4/4] Converting recognizer model with Select TF Ops... ---")
converter_recognizer = tf.lite.TFLiteConverter.from_keras_model(recognizer.model)

# --- This is the necessary configuration ---
converter_recognizer.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TFLite native ops.
    tf.lite.OpsSet.SELECT_TF_OPS    # Enable original TensorFlow ops.
]
# ------------------------------------

tflite_recognizer_model = converter_recognizer.convert()

with open('ocr_recognizer.tflite', 'wb') as f:
    f.write(tflite_recognizer_model)
print("      -> Saved the final ocr_recognizer.tflite")

print("\n--- Conversion complete! This is the definitive model. ---")