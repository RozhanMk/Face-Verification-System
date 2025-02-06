import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import os
from typing import List
from pydantic import BaseModel
from tensorflow.keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity
import random

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class L2NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)
    
    def get_config(self):
        return {'axis': self.axis}

# Load model with custom objects
encoder = tf.keras.models.load_model(
    "encoder-resnet50-margin1.h5",
    custom_objects={'L2NormalizeLayer': L2NormalizeLayer}
)

# Database to store embeddings
EMBEDDINGS_FILE = "face_embeddings.json"

class VerificationRequest(BaseModel):
    name: str = None

def preprocess_image(image: np.ndarray):
    """Process image for model input"""
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

def get_embedding(image: np.ndarray):
    """Generate face embedding using encoder"""
    processed_image = preprocess_image(image)
    return encoder.predict(processed_image, verbose=0)[0]

@app.post("/register/{name}")
async def register_user(
    name: str,
    files: List[UploadFile] = File(...)
):
    """Register new user with multiple images"""
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "r") as f:
                embeddings_db = json.load(f)
        except json.JSONDecodeError:
            embeddings_db = {}  # empty json
    else:
        embeddings_db = {}

    if name in embeddings_db:
        raise HTTPException(status_code=400, detail="User already exists")

    # Process all images
    embeddings = []
    selected_files = random.sample(files, min(5, len(files)))
    for file in selected_files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            continue
            
        embedding = get_embedding(img)
        embeddings.append(embedding.astype(float).tolist())

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid images provided")

    # Store average embedding
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    embeddings_db[name] = avg_embedding

    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings_db, f)

    return {"message": f"User {name} registered successfully"}


@app.post("/verify")
async def verify_user(files: List[UploadFile] = File(...)):
    """Verify user from multiple uploaded images using average of embeddings"""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise HTTPException(status_code=400, detail="No registered users")

    # Process uploaded images and get embeddings
    embeddings = []
    selected_files = random.sample(files, min(5, len(files)))
    for file in selected_files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            continue

        embedding = get_embedding(img)  
        embeddings.append(embedding)

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid images provided")

    avg_embedding = np.mean(embeddings, axis=0)

    # Load stored embeddings
    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings_db = json.load(f)

    max_similarity = -1
    matched_user = None

    for name, stored_embeddings in embeddings_db.items():

        similarity = cosine_similarity([avg_embedding], [stored_embeddings])[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            matched_user = name

    threshold = 0.6
    verified = bool(max_similarity >= threshold)

    return {
        "verified": verified,
        "predicted_name": matched_user if verified else "Unknown",
        "confidence": float(max_similarity)  # Confidence as float
    }





@app.delete("/delete/{name}")
async def delete_user(name: str):
    """Delete a registered user"""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise HTTPException(status_code=400, detail="No registered users")

    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings_db = json.load(f)

    if name not in embeddings_db:
        raise HTTPException(status_code=404, detail="User not found")

    del embeddings_db[name]  # Remove user

    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings_db, f)  # Save updated DB

    return {"message": f"User {name} deleted successfully"}

@app.get("/list_users")
def list_users():
    """Return a list of registered user names."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {"users": []}
    
    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings_db = json.load(f)

    return {"users": list(embeddings_db.keys())}
