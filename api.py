import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import uuid
from pipeline import DiarizationPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Diarization Pipeline API", version="1.0")

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_ENDPOINT = os.getenv("WHISPER_API_ENDPOINT")
LANGUAGE = os.getenv("LANGUAGE", "ur")

UPLOAD_DIR = "/app/uploads"
OUTPUT_DIR = "/app/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {
        "message": "Diarization Pipeline API",
        "version": "1.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    """
    Process audio file through the complete diarization pipeline
    """
    logger.info(f"Received file: {file.filename}")
    
    # Validate file
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid file format. Use WAV, MP3, M4A, or FLAC")
    
    # Generate unique ID
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded file
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"File saved: {input_path}")
    
    try:
        # Create output directory for this job
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Initialize pipeline
        pipeline = DiarizationPipeline(
            hf_token=HF_TOKEN,
            whisper_endpoint=WHISPER_ENDPOINT,
            language=LANGUAGE
        )
        
        # Process audio
        logger.info(f"Starting pipeline for job: {job_id}")
        result = pipeline.process(input_path, job_output_dir)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Pipeline processing failed")
        
        logger.info(f"Pipeline completed for job: {job_id}")
        
        # Cleanup uploaded file
        os.remove(input_path)
        
        return JSONResponse({
            "job_id": job_id,
            "timestamp": timestamp,
            "filename": file.filename,
            "beep_duration_seconds": result["beep_duration"],
            "total_segments": len(result["transcriptions"]),
            "results": result["transcriptions"]
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        
        # Cleanup on error
        if os.path.exists(input_path):
            os.remove(input_path)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)