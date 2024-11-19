import streamlit as st
import requests

# Initialize the Roboflow API client
API_URL = "https://detect.roboflow.com"
API_KEY = "mCrAdZBBO2EnuzdFNUgb"  # Your API key
MODEL_ID = "leaf_disease_detection-yh7jo/1"  # Specify your model ID

# Streamlit UI to interact with the API
def streamlit_app():
    st.title("Leaf Disease Detection")
    st.write("Upload an image to test your model.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Sending the file to the Roboflow API directly
        with st.spinner("Processing..."):
            response = requests.post(
                f"{API_URL}/{MODEL_ID}/infer",  # Roboflow inference URL
                files={"file": uploaded_file.getvalue()},
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success("Inference Complete!")
                st.json(result)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
