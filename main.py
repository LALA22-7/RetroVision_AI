from fastapi import FastAPI, HTTPException
import uvicorn
from vision_processor import process_dashcam_video
import os

app = FastAPI(title="RetroVision AI Backend", version="1.0")

@app.get("/")
def read_root():
    return {"status": "RetroVision AI System is Active", "message": "Ready to process dashcam feeds."}

@app.post("/process-video/")
async def process_video(video_filename: str):
    input_video_path = os.path.join(os.getcwd(), video_filename)
    if not os.path.exists(input_video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    try:
        output_path = process_dashcam_video(input_video_path)
        return {"status": "success", "message": "AI Processing Complete", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)