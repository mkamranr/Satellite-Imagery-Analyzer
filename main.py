# Load model directly
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


processor = CLIPProcessor.from_pretrained("./model")
model = CLIPModel.from_pretrained("./model")

image_path = './images/image3.JPG'

LIST_LABELS = ['agricultural land', 'airplane', 'baseball diamond', 'beach', 'buildings', 'chaparral', 'dense residential area', 'forest', 'freeway', 'golf course', 'harbor', 'intersection', 'medium residential area', 'mobilehome park', 'overpass', 'parking lot', 'river', 'runway', 'sparse residential area', 'storage tanks', 'tennis court']

CLIP_LABELS = [f"A satellite image of {label}" for label in LIST_LABELS]

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

image = Image.open(image_path).convert('RGB')
inputs = processor(text=CLIP_LABELS, images=image, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
prediction = logits_per_image.softmax(dim=1)
confidences = {LIST_LABELS[i]: float(prediction[0][i].item()) for i in range(len(LIST_LABELS))}

percentage_confidences = {
    label: round(confidence * 100, 2)
    for label, confidence in confidences.items()
    if round(confidence * 100, 2) > 1
}

print(percentage_confidences)