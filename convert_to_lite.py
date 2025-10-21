import tensorflow as tf
from keras_ocr.detection import Detector
from keras_ocr.recognition import Recognizer

print("--- [1/4] Re-creating model architectures... ---")
detector = Detector()
recognizer = Recognizer()

print("--- [2/4] Loading saved weights into models... ---")
detector.model.load_weights('detector_weights.h5')
recognizer.model.load_weights('recognizer_weights.h5')

# --- Convert the Detector Model (No changes here) ---
print("--- [3/4] Converting detector model to TFLite... ---")
converter_detector = tf.lite.TFLiteConverter.from_keras_model(detector.model)
tflite_detector_model = converter_detector.convert()
with open('ocr_detector.tflite', 'wb') as f:
    f.write(tflite_detector_model)
print("      -> Saved as ocr_detector.tflite")


# --- Convert the Recognizer Model (with the new, stricter settings) ---
print("--- [4/4] Converting recognizer model to a pure TFLite model... ---")
converter_recognizer = tf.lite.TFLiteConverter.from_keras_model(recognizer.model)

# --- NEW SETTINGS START HERE ---
# Apply default optimizations to help simplify the model graph.
converter_recognizer.optimizations = [tf.lite.Optimize.DEFAULT]

# Define a concrete function to handle dynamic shapes, a common issue.
def representative_dataset():
    # Create a dummy input tensor with a shape the model expects.
    # The recognizer expects images with a height of 31, and a variable width.
    # We provide a representative example with a width of 200.
    for _ in range(1):
        yield [tf.random.normal([1, 31, 200, 1], dtype=tf.float32)]

converter_recognizer.representative_dataset = representative_dataset
# --- NEW SETTINGS END HERE ---

tflite_recognizer_model = converter_recognizer.convert()

with open('ocr_recognizer.tflite', 'wb') as f:
    f.write(tflite_recognizer_model)
print("      -> Saved new ocr_recognizer.tflite")

print("\n--- Conversion complete! ---")