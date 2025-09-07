# 🌿 Plant Disease Recognition System

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)  

A **Streamlit-based web application** for detecting plant leaf diseases using a **Keras deep learning model** trained on the **PlantVillage dataset**. Upload leaf images and get instant predictions with **disease name, cause, and suggested cure**.

---

## 🌱 Features

- Detects **38 plant leaf classes**, including healthy and diseased leaves.
- Shows **disease name, cause, and cure** for each detected disease.
- Simple, **user-friendly interface** via Streamlit.
- Fully offline-compatible model.
- Easy deployment to **Streamlit Community Cloud**.

---

## 📸 Demo

![App Screenshot](https://via.placeholder.com/800x400.png?text=Upload+Leaf+Image)  
*Upload a leaf image and get the disease prediction instantly.*

---

## 📦 Dataset

The model is trained on the **PlantVillage dataset** from Kaggle:  
[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### Classes Included (38):

- Apple Scab, Apple Black Rot, Apple Cedar Apple Rust, Apple Healthy  
- Blueberry Healthy  
- Cherry Powdery Mildew, Cherry Healthy  
- Corn Gray Leaf Spot, Corn Common Rust, Corn Northern Leaf Blight, Corn Healthy  
- Grape Black Rot, Grape Black Measles, Grape Leaf Blight, Grape Healthy  
- Orange Huanglongbing  
- Peach Bacterial Spot, Peach Healthy  
- Pepper Bacterial Spot, Pepper Healthy  
- Potato Early Blight, Potato Healthy, Potato Late Blight  
- Raspberry Healthy  
- Soybean Healthy  
- Squash Powdery Mildew  
- Strawberry Healthy, Strawberry Leaf Scorch  
- Tomato Bacterial Spot, Tomato Early Blight, Tomato Healthy, Tomato Late Blight, Tomato Leaf Mold, Tomato Septoria Leaf Spot, Tomato Spider Mites (Two-spotted), Tomato Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/IamNamanSingh/PlantDiseaseRecognition.git
cd PlantDiseaseRecognition

Install dependencies:

```bash
pip install -r requirements.txt
