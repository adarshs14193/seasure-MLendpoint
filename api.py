from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import uuid
import os
from Inference import run_inference  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Fish Freshness API"}

@app.post("/predict")
async def predict(
    eye_image: UploadFile = File(..., description="Fish eye image"),
    gill_image: UploadFile = File(..., description="Fish gill image")
):
    """
    Predict fish freshness from eye and gill images
    """
    # Save eye image
    eye_id = f"eye_{uuid.uuid4()}.jpg"
    eye_path = os.path.join(UPLOAD_DIR, eye_id)
    
    # Save gill image
    gill_id = f"gill_{uuid.uuid4()}.jpg"
    gill_path = os.path.join(UPLOAD_DIR, gill_id)
    
    try:
        # Save uploaded files
        with open(eye_path, "wb") as buffer:
            shutil.copyfileobj(eye_image.file, buffer)
        
        with open(gill_path, "wb") as buffer:
            shutil.copyfileobj(gill_image.file, buffer)
        
        # Run inference with BOTH images
        result = run_inference(eye_path, gill_path)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Inference failed")
        
        return {
            "status": "success",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded files
        try:
            if os.path.exists(eye_path):
                os.remove(eye_path)
            if os.path.exists(gill_path):
                os.remove(gill_path)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
