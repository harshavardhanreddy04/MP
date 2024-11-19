import streamlit as st
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
from threading import Thread
import uvicorn

# Initialize the Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mCrAdZBBO2EnuzdFNUgb"
)

# FastAPI instance
api = FastAPI()

# Define the FastAPI endpoint
@api.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        # Run inference using the uploaded file
        model_id = "leaf_disease_detection-yh7jo/1"  # Specify your model ID
        result = CLIENT.infer(file.file, model_id=model_id)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Streamlit UI to interact with the API
def streamlit_app():
    st.title("Leaf Disease Detection")
    st.write("Upload an image to test your model.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Sending the file to the FastAPI backend
        with st.spinner("Processing..."):
            import requests
            response = requests.post(
                "http://127.0.0.1:8000/infer",  # FastAPI endpoint
                files={"file": uploaded_file}
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Inference Complete!")
                st.json(result)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Running FastAPI and Streamlit concurrently
def run_apps():
    # Start FastAPI in a separate thread
    def fastapi_thread():
        uvicorn.run(api, host="0.0.0.0", port=8000)

    thread = Thread(target=fastapi_thread)
    thread.daemon = True
    thread.start()

    # Run Streamlit app
    streamlit_app()

if __name__ == "__main__":
    run_apps()
