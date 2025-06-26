from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
from videoProcessor import process_video_to_csv, draw_exoskeleton_on_video

app = FastAPI()

# Allow CORS for all origins (you can restrict as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class NameRequest(BaseModel):
    name: str

@app.post("/api/greet")
def greet_user(request: NameRequest):
    return {"response": f"hello, {request.name}", "detail" : "found" }

@app.post("/api/upload_video")
def upload_video(file: UploadFile = File(...)):
    print(f"[DEBUG] Received file: {file.filename}, content_type: {file.content_type}")
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[DEBUG] Saved uploaded file to: {temp_path}")
    processed_video_path = f"processed_{file.filename}"
    success = draw_exoskeleton_on_video(temp_path, processed_video_path)
    if not success or not os.path.exists(processed_video_path):
        print(f"[ERROR] Processed video not found or empty: {processed_video_path}")
        return {"error": "Processed video not found or empty."}
    print(f"[DEBUG] Returning processed video: {processed_video_path}")
    if not os.path.exists(processed_video_path):
        print(f"[ERROR] File does not exist: {processed_video_path}")
        return {"error": "Processed video file does not exist."}
    if os.path.getsize(processed_video_path) == 0:
        print(f"[ERROR] File is empty: {processed_video_path}")
        return {"error": "Processed video file is empty."}
    # Return just the filename for the frontend to fetch via GET
    return {"filename": os.path.basename(processed_video_path)}

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