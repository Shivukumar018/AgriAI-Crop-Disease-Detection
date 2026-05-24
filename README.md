\# AgriAI 🌱



Smart Crop Disease Detection Using MobileNetV2 Deep Learning.



\## Overview



AgriAI is an AI-powered crop disease detection system that identifies tomato leaf diseases using deep learning and transfer learning techniques.



The system uses MobileNetV2 with TensorFlow and Streamlit for real-time disease prediction from uploaded leaf images.



\---



\## Features



\- Deep Learning based crop disease prediction

\- MobileNetV2 Transfer Learning

\- Real-time image upload

\- AI confidence score visualization

\- Disease recommendation system

\- Streamlit web application



\---



\## Technologies Used



\- Python

\- TensorFlow

\- MobileNetV2

\- OpenCV

\- Streamlit

\- NumPy

\- PIL



\---



\## Dataset



PlantVillage Tomato Leaf Dataset



Classes:

\- Tomato Early Blight

\- Tomato Late Blight

\- Tomato Leaf Mold

\- Tomato Healthy



\---



\## Model Performance



\- Training Accuracy: 97.3%

\- Validation Accuracy: 87.5%



\---



\## Project Architecture



Leaf Image

↓

Image Preprocessing

↓

MobileNetV2 Transfer Learning

↓

Disease Classification

↓

Recommendation System



\---



\## Run Project



```bash

py -3.10 -m streamlit run app.py

