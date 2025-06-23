import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image # Direct import for image module
import numpy as np
from PIL import Image

# --- Configuration ---
# IMPORTANT: Replace 'your_model_directory' with the actual path to your saved model directory.
# Ensure your 'saved_model' directory (e.g., 'my_paddy_disease_model') is in the same
# directory as this Python script, or provide the full path to it.
MODEL_PATH = 'paddyguard_saved_model' # Example: If your model is in 'my_paddy_disease_model' folder
# If your model is saved within a subdirectory, e.g., 'Paddy Disease Detection/paddy_disease_saved_model'
# then update MODEL_PATH accordingly:
# MODEL_PATH = 'Paddy Disease Detection/paddy_disease_saved_model'


# Define the image size your model expects.
# You need to replace these with the actual input shape of your trained CNN model.
# For example, if your model was trained on 224x224 RGB images, set IMAGE_SIZE = (224, 224)
IMAGE_SIZE = (224, 224) # Placeholder: Adjust this to your model's input size
# Updated CLASS_NAMES based on your input
CLASS_NAMES = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# --- Function to load the model ---
# This decorator caches the model so it's loaded only once across reruns of the app
@st.cache_resource
def load_model():
    """Loads the pre-trained TensorFlow SavedModel using tf.saved_model.load.
    This method is robust for SavedModel directories and compatible with newer
    TensorFlow/Keras versions."""
    try:
        # Attempt to load the SavedModel.
        # Ensure the MODEL_PATH correctly points to your saved model directory.
        model = tf.saved_model.load(MODEL_PATH)
        st.success(f"Model loaded successfully from '{MODEL_PATH}'!")
        return model
    except Exception as e:
        # Display an error message if model loading fails.
        st.error(f"Error loading model: {e}")
        st.info("Please ensure your model directory is correctly specified and accessible. "
                "Verify the exact path to your 'saved_model' folder.")
        # Provide detailed traceback for debugging
        st.code(f"Detailed model loading error traceback: {e}", language='python')
        return None

# --- Main Streamlit App ---
def main():
    # Set basic page configuration for the Streamlit app
    st.set_page_config(
        page_title="Paddy Guard", # Title for the browser tab
        page_icon="🌾",                     # Icon for the browser tab
        layout="centered"                   # Centered layout for better aesthetics
    )

    # Display the main title and a brief description
    st.title("🌾 Paddy Guard")
    st.subheader('Developed with ❤️ by Faysal Al Mahmud')
    st.write("Upload an image of a paddy leaf to detect and classify diseases.")

    # Load the model using the cached function
    model = load_model()

    # Proceed only if the model was loaded successfully
    if model is not None:
        # File uploader widget for image selection
        uploaded_file = st.file_uploader(
            "Choose an image...", # User prompt
            type=["jpg", "jpeg", "png"] # Accepted file types
        )

        # Process the uploaded file if one exists
        if uploaded_file is not None:
            # Open the image using PIL (Pillow)
            image_raw = Image.open(uploaded_file)
            # Display the uploaded image in the Streamlit app
            st.image(image_raw, caption="Uploaded Image", use_column_width=True)
            st.write("") # Add some vertical space
            st.write("Classifying...") # Indicate that classification is in progress

            try:
                # Preprocessing the image for model inference:
                # 1. Resize the image to the dimensions expected by the model.
                img = image_raw.resize(IMAGE_SIZE)
                # 2. Convert the PIL image to a NumPy array.
                img_array = image.img_to_array(img)
                # 3. Add a batch dimension (models expect input in batches, even for a single image).
                img_array = np.expand_dims(img_array, axis=0)

                # 4. Normalization: Based on your Colab code, the model likely expects
                #    pixel values in the 0-255 range, or has a Rescaling layer built into it.
                #    Therefore, we are commenting out the explicit 0-1 normalization here
                #    to match your training/testing setup.
                # img_array = img_array / 255.0 # Uncomment if your model expects 0-1 scaled inputs

                # Convert the NumPy array to a TensorFlow tensor, specifying float32 dtype
                input_tensor = tf.constant(img_array, dtype=tf.float32)

                # Make prediction using the loaded SavedModel's 'serving_default' signature.
                # SavedModels loaded with tf.saved_model.load expose concrete functions
                # under model.signatures. 'serving_default' is the common one for inference.
                predictions_output = model.signatures['serving_default'](input_tensor)

                # Extract predictions from the model's output.
                # The output of a serving_default signature is often a dictionary of tensors.
                # 'output_0' is a common key, or it might be the first value in the dictionary.
                # If you encounter issues, uncomment the line below to inspect the keys:
                # st.code(f"Model output keys: {predictions_output.keys()}", language='python')
                predictions = list(predictions_output.values())[0].numpy()

                # --- DEBUGGING OUTPUT (for analyzing model's raw output) ---
                st.subheader("Debugging Info (Expand to see model output)")
                with st.expander("Raw Model Predictions"):
                    st.code(f"Predictions shape: {predictions.shape}", language='python')
                    st.code(f"Predictions values: {predictions}", language='python')
                # --- END DEBUGGING OUTPUT ---

                # Post-processing the predictions:
                # Ensure the predictions array is in the correct shape for argmax.
                # Handles both (1, N) (batch of 1) and (N,) (flattened) shapes.
                if predictions.ndim == 2 and predictions.shape[0] == 1:
                    class_probabilities = predictions[0]
                elif predictions.ndim == 1:
                    class_probabilities = predictions
                else:
                    # If the shape is unexpected, log an error and stop.
                    st.error(f"Unexpected prediction shape: {predictions.shape}. Expected (1, N) or (N,).")
                    st.stop() # Halts script execution

                # Get the index of the highest probability (predicted class)
                predicted_class_index = np.argmax(class_probabilities)
                # Calculate confidence as a percentage
                confidence = np.max(class_probabilities) * 100

                # Get the human-readable class name
                predicted_class_name = CLASS_NAMES[predicted_class_index]

                # Display the prediction and confidence
                st.success(f"Prediction: **{predicted_class_name.replace('_', ' ').title()}**")
                st.write(f"Confidence: **{confidence:.2f}%**")

            except Exception as e:
                # Catch any errors during the prediction process
                st.error(f"An error occurred during image classification: {e}")
                st.info("Please verify your `IMAGE_SIZE` and ensure the model's output "
                        "structure (e.g., key 'output_0') is correctly handled.")
                # Display detailed error for debugging
                st.code(f"Detailed prediction error traceback: {e}", language='python')
    else:
        # Message displayed if the model failed to load
        st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure your model directory exists and your TensorFlow installation is correct.")

if __name__ == "__main__":
    main()
