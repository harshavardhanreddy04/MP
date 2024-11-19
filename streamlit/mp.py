import streamlit as st
import requests
from PIL import Image, ImageDraw

# Roboflow API URL and API key
API_URL = "https://detect.roboflow.com"
API_KEY = "mCrAdZBBO2EnuzdFNUgb"  # Your API key
MODEL_ID = "leaf_disease_detection-yh7jo/1"  # Your model ID

# Streamlit UI to interact with the API
def streamlit_app():
    st.title("Leaf Disease Detection with Object Detection")
    st.write("Upload an image to test your model for object detection.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Sending the file to the Roboflow API for object detection
        with st.spinner("Processing..."):
            try:
                # Ensure the endpoint and method are correct for the object detection API
                response = requests.post(
                    f"{API_URL}/{MODEL_ID}/infer",  # Roboflow inference URL (Ensure this is correct)
                    files={"file": uploaded_file.getvalue()},
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )

                # Check if the response status is OK (200)
                if response.status_code == 200:
                    result = response.json()

                    # Object detection results
                    predictions = result.get('predictions', [])
                    if predictions:
                        st.success("Inference Complete!")

                        # Draw bounding boxes on the image
                        img = Image.open(uploaded_file)
                        draw = ImageDraw.Draw(img)

                        for pred in predictions:
                            # Bounding box coordinates
                            x_min, y_min, x_max, y_max = pred['x'], pred['y'], pred['x2'], pred['y2']
                            label = pred['class']
                            confidence = pred['confidence']

                            # Draw the bounding box and label
                            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                            draw.text((x_min, y_min - 10), f"{label} ({confidence:.2f})", fill="red")

                        # Display the image with bounding boxes
                        st.image(img, caption="Processed Image with Object Detection", use_container_width=True)

                        # Show raw prediction data (if needed)
                        st.json(result)

                    else:
                        st.warning("No objects detected.")
                else:
                    # Log and display the error response
                    error_message = response.json()
                    st.error(f"Error: {error_message.get('error', 'Unknown error')}")
                    st.json(error_message)  # Show full error message for debugging

            except Exception as e:
                # Catching any exception and displaying it
                st.error(f"An error occurred: {str(e)}")

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
