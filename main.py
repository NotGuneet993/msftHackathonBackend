from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow CORS for all origins (you can restrict as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NameRequest(BaseModel):
    name: str

@app.post("/api/greet")
def greet_user(request: NameRequest):
    return {"response": f"hello, {request.name}", "detail" : "found" }

@app.post("/api/upload_video")
def upload_video(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type, "detail": "file received"}