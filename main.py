from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class NameRequest(BaseModel):
    name: str

@app.post("/api/greet")
def greet_user(request: NameRequest):
    return {"response": f"hello, {request.name}", "detail" : "found" }