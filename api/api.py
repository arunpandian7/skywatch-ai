import io
from fastapi import FastAPI, UploadFile, File
import skywatchai.SkywatchAI as skai
import skywatchai.SkywatchDB as skdb

from utils import process_image
from starlette.responses import StreamingResponse

app = FastAPI(title="skywatch.ai", debug=True)
faceDB, nameMap = skdb.load_db('database/')

@app.get('/')
async def root():
    return {"message", "SkywatchAI API"}

@app.post('/detections')
async def get_detections(img : UploadFile = File(...)):
    p_img = process_image(img)
    detected = skai.detect_faces(p_img)
    return StreamingResponse(io.BytesIO(detected.tobytes()), media_type="image/png")

@app.post('/verifications')
async def get_verifications(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    img1 = process_image(img1)
    img2 = process_image(img2)
    result = skai.compare(img1, img2)
    return {
        "verified": result
    }

@app.post('/recognitons')
async def get_recognitions(img: UploadFile = File(...)):
    img = process_image(img)
    annotImage = skai.find_people(img, faceDB, nameMap)
    return StreamingResponse(io.BytesIO(annotImage.tobytes()), media_type="image/png")
    
@app.post('/build-db')
async def build_db(path: str):
    skdb.build_db(path, path)
    return "Skywatch FaceDB successfully built"
