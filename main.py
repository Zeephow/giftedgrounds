from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os

app = FastAPI()

# Allow PHP frontend to call FastAPI (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust this to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Whisper model once
model = whisper.load_model("base")  # "small" / "medium" / "large" also possible

@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    result = model.transcribe(file_path)
    os.remove(file_path)
    
    return {"transcription": result["text"]}
