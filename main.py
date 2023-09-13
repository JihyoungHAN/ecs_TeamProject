from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image 
import numpy as np 
import io 
import os
import yolo_TrafficLight_image_func as yolo

app = FastAPI()

#setting for using HTML Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def upload_image_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/")
async def upload_file(file: UploadFile, request: Request): 
    image_data = await file.read()
    with Image.open(io.BytesIO(image_data)) as image: 
        image = np.array(image)
        result = yolo.classify(image)
    if result == 100: 
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    elif result == 110: 
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    else: 
        return templates.TemplateResponse("index.html", {"request": request, "result": result})




if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)
