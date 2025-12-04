import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
import os
import uvicorn
from typing import List

app = FastAPI(title="Sign Language AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Sign Language AI Model...")

# Initialize as None, load in startup event
model = None
class_mapping = None
model_info = None

@app.on_event("startup")
async def load_model():
    global model, class_mapping, model_info
    try:
        model = tf.keras.models.load_model('sign_language_model_final.h5')
        with open('class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        print("Model loaded successfully")
        print(f"Model configured for {len(class_mapping)} sign language classes")
        print(f"Classes: {list(class_mapping.keys())}")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Don't raise here, let the app start but mark as unhealthy
        print("WARNING: Model failed to load, API will run in degraded mode")

def preprocess_data(input_data: List[float]) -> np.ndarray:
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    data_array = np.array(input_data, dtype=np.float32)
    
    if len(data_array.shape) > 1:
        data_array = data_array.flatten()
    
    target_size = model_info['feature_size']
    if len(data_array) > target_size:
        data_array = data_array[:target_size]
    elif len(data_array) < target_size:
        padded = np.zeros(target_size, dtype=np.float32)
        padded[:len(data_array)] = data_array
        data_array = padded
    
    data_array = (data_array - np.mean(data_array)) / (np.std(data_array) + 1e-8)
    
    return data_array.reshape(1, -1)

@app.get("/")
async def root():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "message": "Sign Language AI API is running",
        "model_status": model_status,
        "status": "active"
    }

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "degraded", "model_loaded": False, "message": "Model failed to load"}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(data: List[float]):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Service is starting or failed to load model.")
        
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        processed_data = preprocess_data(data)
        
        predictions = model.predict(processed_data)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        class_name = list(class_mapping.keys())[predicted_class_idx]
        
        return {
            "success": True,
            "predicted_class": predicted_class_idx,
            "class_name": class_name,
            "confidence": confidence,
            "all_predictions": predictions[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "input_shape": model_info['input_shape'],
        "num_classes": model_info['num_classes'],
        "classes": model_info['classes'],
        "feature_size": model_info['feature_size']
    }

# This block won't be executed by gunicorn, but kept for local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
