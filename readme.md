# 🧠 Brain Tumor Detection Using Deep Learning

This repository contains a **Deep Learning–based Brain Tumor Detection system** that classifies brain MRI images to identify the presence and type of brain tumors. The model is designed to assist medical professionals and researchers by providing fast and accurate predictions using computer vision techniques.

---

## 📌 Project Overview

Brain tumors are one of the most critical medical conditions requiring early and accurate diagnosis. Manual diagnosis from MRI scans is time-consuming and subject to human error. This project leverages **( Transfer Learning)** to automatically analyze MRI images and classify them into tumor categories.

The system:

* Takes MRI images as input
* Preprocesses and normalizes data
* Uses a trained deep learning model
* Outputs the predicted tumor class with confidence

---

## 🧪 Tumor Classes

Depending on the dataset used, the model supports classification into the following categories:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor


---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras or PyTorch
* **Model Architecture:**  ResNet (Transfer Learning)
* **Image Processing:** OpenCV, PIL
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---



## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/kg0420/Brain-Tumor-Detection.git
cd brain-tumor-detection
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training


Training includes:

* Image augmentation
* Train–validation split
* Model checkpointing
* Accuracy and loss monitoring

---

## 🔍 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example confusion matrix:

```
True Positives | False Positives
False Negatives| True Negatives
```

---

## 🚀 Prediction / Inference

To make predictions on new MRI images:

```bash
python src/predict.py --image path_to_image.jpg
```

Output:

```
Prediction: Glioma Tumor
Confidence: 97.3%
```

---

## 🌐 Deployment (Optional)

The model can be deployed using:

* Flask / FastAPI (Backend)

Example:

```bash
python app.py
```

---

## 📊 Results

* Train Accuracy  : 99.72 %
* Test Accuracy   : 95.44 %
* Precision Score : 95.44 %
* Recall Score    : 95.44 %


---


---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Public Brain MRI Datasets
* TensorFlow / PyTorch Community
* Medical Imaging Research Papers

---

## 👨‍💻 Author

**Krish Gupta**
Computer Engineering Student | AI & ML Enthusiast

---

⭐ If you found this project helpful, don’t forget to star the repository!
