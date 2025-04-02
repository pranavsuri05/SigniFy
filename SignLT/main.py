from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from train_model import predict_sign  # Import your AI model function

app = FastAPI()

@app.post("/translate-sign")
async def translate_sign(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run AI model on frame
    translated_text = predict_sign(frame)

    return {"text": translated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
