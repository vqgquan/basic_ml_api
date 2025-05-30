from typing import Annotated
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from app.model import FashionClassifier


MODEL_PATH = 'app/model.pth'
model = FashionClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.get("/")
def read_root():
    return {"message":"AlexNet based model"}


@app.post("/create-files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/upload-file/")
async def create_upload_file(file: UploadFile):
    # Save the uploaded file temporarily
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Run inference
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    output = model(image)
    _, predicted = torch.max(output, 1)
    
    return {"filename": file.filename, "prediction": predicted.item()}