from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

processor = CLIPProcessor.from_pretrained("./model")
model = CLIPModel.from_pretrained("./model")

LIST_LABELS = ['agricultural land', 'airplane', 'baseball diamond', 'beach', 'buildings', 'chaparral',
               'dense residential area', 'forest', 'freeway', 'golf course', 'harbor', 'intersection',
               'medium residential area', 'mobilehome park', 'overpass', 'parking lot', 'river', 'runway',
               'sparse residential area', 'storage tanks', 'tennis court']

CLIP_LABELS = [f"A satellite image of {label}" for label in LIST_LABELS]

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


# Split the image into 200x200 sub-images and return their bounding boxes
def split_image(image, sub_image_size=(200, 200)):
    width, height = image.size
    sub_images = []
    bounding_boxes = []

    # Calculate the number of rows and columns needed
    num_rows = height // sub_image_size[1]
    num_cols = width // sub_image_size[0]

    for row in range(num_rows + 1):
        for col in range(num_cols + 1):
            left = col * sub_image_size[0]
            upper = row * sub_image_size[1]
            right = min(left + sub_image_size[0], width)
            lower = min(upper + sub_image_size[1], height)

            # Crop sub-image and store its bounding box
            sub_image = image.crop((left, upper, right, lower))
            sub_images.append(sub_image)
            bounding_boxes.append((left, upper, right, lower))

    return sub_images, bounding_boxes


# Function to analyze each sub-image using the CLIP model
def analyze_sub_images(sub_images, processor, model, device, labels):
    results = []

    for sub_image in sub_images:
        inputs = processor(text=labels, images=sub_image, return_tensors="pt", padding=True).to(device)
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

        results.append(percentage_confidences)

    return results


# Function to recombine sub-images into one image and draw bounding boxes with labels
def recombine_images_with_bounding_boxes(original_image, sub_images, bounding_boxes, labels_with_confidence):
    # Create a copy of the original image to draw bounding boxes
    combined_image = original_image.copy()
    draw = ImageDraw.Draw(combined_image)

    # Define font for label text
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    # Loop over sub-images, bounding boxes, and labels
    for sub_image, bbox, label_conf in zip(sub_images, bounding_boxes, labels_with_confidence):
        # Draw the bounding box
        left, upper, right, lower = bbox
        draw.rectangle([left, upper, right, lower], outline="red", width=1)

        # Get the label with the highest confidence
        if label_conf:
            highest_label = max(label_conf, key=label_conf.get)
            highest_confidence = label_conf[highest_label]
            label_text = f"{highest_label} ({highest_confidence}%)"

            # Use textbbox to calculate text size and position
            text_bbox = draw.textbbox((left, upper), label_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

            # Adjust position so text is above the bounding box
            text_position = (left, upper - text_height if upper - text_height > 0 else upper + 2)
            draw.rectangle([text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                           fill="black")
            draw.text(text_position, label_text, fill="white", font=font)

    return combined_image

# Load image
image_path = './images/image3.JPG'
image = Image.open(image_path).convert('RGB')

# Split image into sub-images
sub_images, bounding_boxes = split_image(image, (300, 300))

# Analyze each sub-image
results = analyze_sub_images(sub_images, processor, model, device, CLIP_LABELS)

# # Combine results with bounding boxes
# for idx, (result, box) in enumerate(zip(results, bounding_boxes)):
#     print(f"Sub-image {idx} (box: {box}): {result}")

# Combine the sub-images back into the original image and draw bounding boxes with labels
final_image = recombine_images_with_bounding_boxes(image, sub_images, bounding_boxes, results)

# Save or display the final image
final_image.show()  # To display the image
final_image.save('./output_images/combined_with_labels.jpg')  # To save the image
