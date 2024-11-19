import streamlit as st
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import uvicorn
from threading import Thread

# Initialize the Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mCrAdZBBO2EnuzdFNUgb"
)

# FastAPI App
api = FastAPI()

@api.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        # Run inference using the uploaded file
        model_id = "leaf_disease_detection-yh7jo/1"  # Specify your model ID
        result = CLIENT.infer(file.file, model_id=model_id)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Streamlit App
def start_streamlit():
    st.title("Leaf Disease Detection")
    st.write("Upload an image via the Flutter app to get results here.")

# Run FastAPI and Streamlit together
def run_apps():
    # Start Streamlit in a separate thread
    thread = Thread(target=lambda: st._run_app(start_streamlit, "streamlit run"))
    thread.start()
    
    # Run FastAPI app
    uvicorn.run(api, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_apps()
