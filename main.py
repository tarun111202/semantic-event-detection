!pip install torch torchvision open_clip_torch opencv-python matplotlib
import torch

print("GPU Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from google.colab import files

uploaded = files.upload()
import open_clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

tokenizer = open_clip.get_tokenizer('ViT-B-32')

model = model.to(device)
model.eval()

print("Model loaded on:", device)
text_prompts = [
    "a person walking",
    "a vehicle stopping",
    "a crowded scene"
]

text_tokens = tokenizer(text_prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print("Text features ready.")
import os
print(os.listdir())
video_path = "input_video.mp4"
import cv2

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")

cap.release()
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
processed_frames = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 10th frame
    if frame_count % 10 != 0:
        continue

    processed_frames += 1

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_event = text_prompts[similarity.argmax().item()]

    print(f"Frame {frame_count}: {predicted_event}")

end_time = time.time()

cap.release()

print("Total Frames Read:", frame_count)
print("Total Frames Processed:", processed_frames)
print("Total Inference Time:", end_time - start_time)
torch.save(model.state_dict(), "original_model.pth")

import os
size_mb = os.path.getsize("original_model.pth") / (1024 * 1024)
print("Original Model Size (MB):", size_mb)
import torch.quantization

model_cpu = model.to("cpu")

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("Quantization completed.")
torch.save(quantized_model.state_dict(), "quantized_model.pth")

size_q_mb = os.path.getsize("quantized_model.pth") / (1024 * 1024)
print("Quantized Model Size (MB):", size_q_mb)
cap = cv2.VideoCapture(video_path)

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 30 != 0:
        continue

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_features = quantized_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.cpu().T).softmax(dim=-1)
        predicted_event = text_prompts[similarity.argmax().item()]

    print(f"[Quantized] Frame {frame_count}: {predicted_event}")

end_time = time.time()
print("Quantized Inference Time:", end_time - start_time)

cap.release()
print("----- PERFORMANCE COMPARISON -----")
print("Original Model Size (MB):", size_mb)
print("Quantized Model Size (MB):", size_q_mb)



