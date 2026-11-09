from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
from app.ml.inference import SentencePredictor

router = APIRouter()

@router.websocket("/ws/isl-to-text")
async def isl_to_text(websocket: WebSocket):
    await websocket.accept()
    predictor = SentencePredictor()
    print("Client connected")

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            predictor.add_frame(frame)
            word, confidence, buffer_len = predictor.predict()

            await websocket.send_json({
                "prediction": word,
                "confidence": round(confidence, 3),
                "buffer_progress": f"{buffer_len}/30"
            })

    except WebSocketDisconnect:
        print("Client disconnected")