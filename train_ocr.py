import tensorflow.keras as keras
import keras_ocr
import os

# --- 1. Load the Pre-trained Pipeline ---
# This only needs to be done once
pipeline = keras_ocr.pipeline.Pipeline()

# --- 2. Define the path to your receipt images ---
image_dir = 'receipt_images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

# --- 3. Process images ONE BY ONE in a loop ---
if image_paths:
    print(f"Found {len(image_paths)} image(s) to process.")
    
    # Loop through each image path
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
        
        # Load just the single image
        image_list = [keras_ocr.tools.read(image_path)]
        
        # Run prediction on that single image
        prediction_groups = pipeline.recognize(image_list)
        
        # Print the recognized text for the current image
        print("--- Recognized Text: ---")
        for text, box in prediction_groups[0]:
            print(text)

    # --- 4. Save the model weights for conversion ---
    # This is done after processing all images
    detector = pipeline.detector
    recognizer = pipeline.recognizer

    detector.model.save_weights('detector_weights.h5')
    recognizer.model.save_weights('recognizer_weights.h5')
    
    print("\n--- All images processed. Model weights saved successfully! ---")

else:
    print(f"No images found in '{image_dir}' folder. Please add some receipt images to test the model.")