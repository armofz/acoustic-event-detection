"""MVP server that runs the detector and sends what it hears to every open page through a WebSocket.
Only events, probabilities and the spectrogram leave the detector.
Run from the repo root: python src/app.py
Then open http://localhost:8000 on this computer, or http://<computer's IP>:8000 on a phone in the same Wi-Fi.
"""
import asyncio
import csv
import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import Detector
from make_logmel import MANIFEST


pages = set()    # WebSockets of the open pages
loop = None    # event loop of the server, to send messages from the detector thread


def send_to_pages(message):
    """Called by the detector for every frame, alert and status change."""
    if loop is not None:
        asyncio.run_coroutine_threadsafe(send_all(json.dumps(message)), loop)


async def send_all(text):
    for page in list(pages):
        try:
            await page.send_text(text)
        except Exception:
            pages.discard(page)


detector = Detector(on_message=send_to_pages)


@asynccontextmanager
async def lifespan(app):
    global loop
    loop = asyncio.get_running_loop()
    yield
    detector.stop_mic()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket(page: WebSocket):
    """Each page receives the detector messages and sends commands (listen, model, settings, simulate)."""
    await page.accept()
    pages.add(page)
    await page.send_text(json.dumps(detector.status()))
    try:
        while True:
            command = json.loads(await page.receive_text())
            await asyncio.to_thread(detector.handle, command)    # in a thread, loading a model takes a few seconds
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        pages.discard(page)


@app.get("/clip/{clip_id}")
def clip(clip_id: str):
    """Audio of a test clip, so the page can play the simulated sound. Only test clips can be downloaded."""
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["clip_id"] == clip_id and row["split"] == "test":
                return FileResponse(row["path"])
    raise HTTPException(status_code=404)


app.mount("/", StaticFiles(directory="web", html=True))    # web/index.html


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    # 0.0.0.0: reachable from other devices in the same network