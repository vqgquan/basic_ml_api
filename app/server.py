from typing import Annotated
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from app.model import FashionClassifier
from fastapi.middleware.cors import CORSMiddleware


MODEL_PATH = 'app/model.pth'
LABELS = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}



model = FashionClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


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
    
    return {"filename": file.filename, "prediction": predicted.item(), "label" : LABELS[predicted.item()]}