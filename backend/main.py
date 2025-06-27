from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from videoProcessor import process_video_to_csv, draw_exoskeleton_on_video
from model_utils import SquatFormPredictor
from pydantic import BaseModel

app = FastAPI()

# Initialize the model predictor
try:
    model_predictor = SquatFormPredictor(model_path="model/lstm_squat_model.pt")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_predictor = None

class NameRequest(BaseModel):
    name: str

@app.post("/api/greet")
def greet_user(request: NameRequest):
    return {"response": f"hello, {request.name}", "detail": "found"}

@app.post("/api/upload_video")
def upload_video(file: UploadFile = File(...)):
    print(f"[DEBUG] Received file: {file.filename}, content_type: {file.content_type}")
    
    # Step 1: Save uploaded video temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[DEBUG] Saved uploaded file to: {temp_path}")
    
    try:
        # Step 2: Generate video overlay with pose detection
        processed_video_path = f"processed_{file.filename}"
        success = draw_exoskeleton_on_video(temp_path, processed_video_path)
        
        if not success or not os.path.exists(processed_video_path):
            print(f"[ERROR] Processed video not found or empty: {processed_video_path}")
            return {"error": "Failed to process video with pose detection."}
        
        # Step 3: Create CSV of the frames with landmarks
        csv_filename = f"landmarks_{os.path.splitext(file.filename)[0]}.csv"
        csv_path = os.path.join("outputs", csv_filename)
        process_video_to_csv(temp_path, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV file not created: {csv_path}")
            return {"error": "Failed to extract landmarks to CSV."}
        
        # Step 4: Load CSV into model and get prediction
        try:
            if model_predictor is None:
                raise Exception("Model not loaded")
            prediction_result = model_predictor.predict(csv_path)
            print(f"[DEBUG] Model prediction: {prediction_result}")
        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}")
            prediction_result = {
                "error": f"Model prediction failed: {str(e)}",
                "predicted_label": "unknown",
                "confidence": 0.0
            }
        
        # Validate processed video
        if not os.path.exists(processed_video_path):
            print(f"[ERROR] File does not exist: {processed_video_path}")
            return {"error": "Processed video file does not exist."}
        if os.path.getsize(processed_video_path) == 0:
            print(f"[ERROR] File is empty: {processed_video_path}")
            return {"error": "Processed video file is empty."}
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"[WARNING] Could not remove temp file {temp_path}: {e}")
        
        # Return overlay video filename and prediction results
        return {
            "filename": os.path.basename(processed_video_path),
            "prediction": prediction_result
        }
    
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        # Clean up on error
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
        return {"error": f"Video processing pipeline failed: {str(e)}"}

@app.get("/videos/{filename}")
def get_video(filename: str):
    video_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(video_path):
        return {"error": "Video not found."}
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )

@app.get("/api/model/status")
def get_model_status():
    """Check if the model is loaded and working"""
    try:
        if model_predictor.model is None:
            return {"status": "error", "message": "Model not loaded"}
        return {
            "status": "ready",
            "message": "Model loaded successfully",
            "device": str(model_predictor.device),
            "target_length": model_predictor.target_len,
            "supported_classes": list(model_predictor.model.__class__.__dict__.get('LABEL_MAP', {}).values()) or [
                "good", "bothknees", "buttwink", "halfsquat", "leanforward", "leftknee", "rightknee"
            ]
        }
    except Exception as e:
        return {"status": "error", "message": f"Model status check failed: {str(e)}"}

@app.post("/api/predict_csv")
def predict_from_csv(csv_filename: str):
    """Test prediction on an existing CSV file"""
    try:
        csv_path = os.path.join("outputs", csv_filename)
        if not os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_filename}"}
        
        prediction_result = model_predictor.predict(csv_path)
        return {"prediction": prediction_result, "csv_file": csv_filename}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
