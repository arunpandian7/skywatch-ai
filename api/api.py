from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="skywatch.ai")

@app.get('/')
async def root():
    return {"message", "Hello World"}

@app.post('/detections')
async def get_detections(img : UploadFile = File(...)):
    return {"response", "got the file"+img.filename}
