from PIL import Image
import torch

image = Image.open("test.jpg")

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

pred = torch.argmax(logits, dim=1).item()
print(model.config.id2label[pred])