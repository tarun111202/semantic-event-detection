# semantic-event-detection
Semantic Event Detection using Optimized Vision–Language Model (OpenCLIP) with Dynamic Quantization for real-time inference.
## Note

The sample video file is not included due to GitHub file size limitations.
You can use any .mp4 video file and update the path inside main.py.
# Semantic Event Detection with Optimized Vision–Language Model

## 📌 Overview
This project performs semantic event detection from video using the **OpenCLIP Vision–Language Model (ViT-B/32)** and demonstrates model optimization using **dynamic quantization** for improved inference efficiency.

The system processes a video file, encodes video frames and text prompts into embeddings, and predicts semantic events based on cosine similarity.

---

## 🚀 Features

✔ Zero-shot event detection  
✔ Uses OpenCLIP (ViT-B/32)  
✔ Dynamic INT8 quantization for model optimization  
✔ CPU-side inference  
✔ Clear performance comparison  

---

## 🎯 Events Detected

The system can detect the following semantic events:

- A person walking  
- Vehicle stopping  
- Crowded scene

---

## 🏃‍♂️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/tarun111202/semantic-event-detection.git
Navigate to project directory: cd semantic-event-detection
Install dependencies: pip install -r requirements.txt
Run the script:python main.py




