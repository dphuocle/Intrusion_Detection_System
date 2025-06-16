
# 🛡️ Intrusion Detection System (IDS) using CNN + LSTM model

A deep learning–based **Intrusion Detection System** (IDS) capable of identifying and classifying network traffic as either benign or one of several types of cyberattacks. This project uses a hybrid **CNN + LSTM neural network** trained on the **CSE-CIC-IDS2018** dataset and deployed as a web app with **Streamlit**.

---

## 🌐 Live Demo

👉 [Try it on Streamlit](https://intrusiondetectionsystem-2301.streamlit.app)

---

## 📌 Introduction

As cyber threats continue to grow in scale and complexity, traditional rule-based intrusion detection systems (IDS) struggle to keep up with evolving attack patterns. This project presents a modern, data-driven IDS that leverages the power of deep learning to automatically detect and classify malicious behavior in network traffic.

We train and deploy a hybrid **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** model capable of classifying flow-level network traffic into 8 different classes, including various forms of **Bot**, **SSH_BruteForce**, **DDoS_HOIC**, **DoS_Hulk**, **FTP_BruteForce**, **DDoS_LOIC_HTTP**, **DoS_GoldenEyeand** and **benign** behavior.

---

## 🧠 Model Overview

The architecture combines the strengths of CNNs (feature extraction) and LSTMs (temporal dependencies), making it well-suited for analyzing flow-based time-dependent network traffic.

![image](https://github.com/user-attachments/assets/fa086933-72d1-4e91-887f-5f44f947b35e)
*Source : A hybrid CNN+LSTM-based intrusion detection system for industrial IoT networks* 

### 📐 Architecture
```
Input [Batch, Features]
→ Conv1D → ReLU → MaxPool
→ Conv1D → ReLU → MaxPool
→ LSTM (sequence modeling)
→ Fully Connected → Dropout → FC → Softmax (8 classes)
```

---

## 🏋️ Training Setup

| Parameter        | Value                                 |
|------------------|---------------------------------------|
| Dataset          | Subset of CSE-CIC-IDS2018             |
| Features         | 78 numerical network flow features    |
| Labels           | 8 classes (Benign + 7 attack types)   |
| Split Ratio      | 80% train / 10% val / 10% test        |
| Loss             | CrossEntropyLoss                      |
| Optimizer        | Adam (learning rate = 1e-4)           |
| Epochs           | 20                                    |
| Batch Size       | 64                                    |

---

## 📊 Results

- ✅ **Validation Accuracy**: `99.95%` 
- ✅ **Test Accuracy**: `99.97%`

### 📈 Training Curves

![Loss and Accuracy over Epochs](https://github.com/user-attachments/assets/77678104-4d7e-46b7-b964-b523ed97970c)

### 🔍 Confusion Matrix

![Confusion Matrix](https://github.com/user-attachments/assets/f164c0bf-4e6a-48df-ab0d-825efe0392d3)


The model demonstrates high performance across all attack classes, with strong generalization and minimal overfitting.

---

## ⚙️ Deployment

The trained model (`model_hybrid_cnn_lstm.pt`) is integrated into a **Streamlit web app** allowing users to upload `.csv` files of flow-based traffic and receive predictions (with confidence scores) for each row.

### 🔧 How to Run Locally

```bash
git clone https://github.com/dphuocle/Intrusion_Detection_System.git
cd Intrusion_Detection_System
pip install -r requirements.txt
streamlit run deploy.py
```

---

## 📁 Project Structure

```
📦 Intrusion_Detection_System/
├── deploy.py                  # Streamlit UI
├── model-CNN-LSTM.ipynb       # Code to train the model
├── model_hybrid_cnn_lstm.pt   # Trained model
├── requirements.txt           # Python dependencies
├── test.csv                   # Sample test data
├── predictions.csv            # Sample predictions from the test data
├── README.md
```

---

## 📦 Requirements

- Python 3.9+
- PyTorch
- Streamlit
- scikit-learn
- pandas, numpy, matplotlib

---

## 👤 Author

**LE Doan Phuoc**  
AI Intern at MS4ALL and CS Student at INSA Centre Val de Loire

🔗 [LinkedIn](https://www.linkedin.com/in/dphuocle/)
