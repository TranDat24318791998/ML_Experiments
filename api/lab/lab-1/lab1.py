from fastapi import FastAPI
import uvicorn
from typing import Literal
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: Literal["oke", "fail", "error", "starting"]

# Start a simple fastapi app
app = FastAPI(title="API lab 1", version="1.0.0")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="oke")

if __name__ == "__main__":
    uvicorn.run("lab1:app", host="127.0.0.1", port=8081, reload=True)