# Tech-phantoms Intel AI Hackathon Prototype
# Project Title: Emotion Detection in Facial Images

## Idea:
Our project aims to develop a deep learning model for emotion detection in facial images. By leveraging state-of-the-art image classification techniques, we intend to accurately classify emotions depicted in facial images into predefined categories.

## Purpose:
The purpose of our project is to provide a tool that can analyze facial expressions and infer the corresponding emotions. This technology can be applied in various domains such as market research, healthcare, and human-computer interaction. For instance, it can help companies understand customer sentiments towards their products or services, assist therapists in assessing emotional states of patients, and enhance user experience in interactive systems.

## Benefits:
- **Enhanced Decision Making:** Businesses can make data-driven decisions by understanding customer emotions and preferences.
- **Improved Healthcare:** Therapists can utilize the tool to monitor patients' emotional well-being and provide timely interventions.
- **Enhanced User Experience:** Applications can personalize user experiences based on detected emotions, leading to higher user engagement and satisfaction.

## Equipment Used:
- **Model:** We used a pre-trained deep learning model for image classification, specifically "dima806/facial_emotions_image_detection".
- **Programming Language:** Python was used for coding the project.
- **Libraries:** We utilized the Transformers library from Hugging Face for model loading and inference, along with other standard libraries such as PyTorch and torchvision.
- **Data:** The model was trained on a dataset consisting of facial images labeled with corresponding emotions.

## Sources:
- **Model:** The deep learning model used in this project was obtained from the Hugging Face model hub (https://huggingface.co/dima806/facial_emotions_image_detection).
- **Data:** The training data for the model was sourced from publicly available datasets such as the FER2013 dataset.
- **Libraries:** We referred to the official documentation and resources provided by the Transformers library and PyTorch community for implementation guidance.

# Emotion Detection in Facial Images - Code Explanation

## Code Overview:

The provided code implements an emotion detection system for facial images using a pre-trained deep learning model. The system processes a video file frame by frame, predicts the emotions depicted in each frame, and logs the results.

### Code 1: Without Intel Optimization

The first code block initializes the emotion detection pipeline and processes each frame of the video without Intel optimization. Here's how it works:

- **Initialization:** 
  - The `EmotionDetector` class loads the pre-trained image classification model (`dima806/facial_emotions_image_detection`) from the Hugging Face model hub and sets up the preprocessing pipeline.
  - The `VideoProcessor` class initializes the video processing pipeline and sets up the video capture.
  
- **Frame Processing:**
  - The video frames are read sequentially.
  - For every 300 frames, the frame is converted to a PIL image and fed into the emotion detection model.
  - The predicted emotion for each frame is printed and saved to disk.

### Code 2: With Intel Optimization

The second code block optimizes the emotion detection pipeline using Intel Extension for PyTorch (IPEX). Here's how it differs from the first code:

- **Optimization:** 
  - The model loading is optimized using IPEX's `optimize` function to leverage Intel's hardware capabilities for improved performance.
  - The preprocessing step is optimized to run on the CPU, utilizing IPEX's capabilities for enhanced efficiency.

- **Frame Processing:**
  - The video frames are processed in a similar manner to Code 1, but with the added benefit of Intel optimization for improved performance.

## Intel Optimization:

Intel Extension for PyTorch (IPEX) is a set of extensions for PyTorch that improves performance on Intel CPUs. It optimizes various aspects of the deep learning pipeline, including model loading, preprocessing, and inference. Here's how IPEX enhances the code:

- **Model Optimization:** 
  - IPEX optimizes the model loading process, ensuring efficient utilization of Intel's hardware resources.
  
- **Preprocessing Optimization:**
  - IPEX optimizes the image preprocessing step to run efficiently on the CPU, improving overall pipeline performance.
  
- **Inference Optimization:**
  - IPEX enhances inference performance, resulting in faster predictions and reduced execution time.

By leveraging Intel's hardware capabilities through IPEX, the code achieves improved efficiency and performance, making it suitable for real-time applications.

For further information about IPEX and its capabilities, refer to the official Intel documentation and resources.

