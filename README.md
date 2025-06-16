# 🛡️ Intrusion Detection System (IDS) – Deep Learning Powered

A real-time, deployable Intrusion Detection System (IDS) that detects malicious network flows using a hybrid **CNN + LSTM** deep learning model.  
Built with **Streamlit** for an intuitive web interface and deployed to the cloud.

![image](https://github.com/user-attachments/assets/bc403870-93b6-40d6-a3d7-613307a80ab0)

---

## 🚀 Demo

👉 [Click to Try](https://intrusiondetectionsystem-2301.streamlit.app)

---

## 🧠 How It Works

- Upload any `.csv` file containing network flow records from datasets like CSE-CIC-IDS2018
- The model will automatically:
  - Clean and normalize your data
  - Run predictions using a trained CNN + LSTM model
  - Return one of the type of attack label with confidence scores

---

## 🧰 Tech Stack

| Component      | Tool              |
|----------------|------------------|
| Web Interface  | [Streamlit](https://streamlit.io) |
| Model          | PyTorch (CNN + LSTM) |
| Dataset        | CSE-CIC-IDS2018 (Botnet, DDoS, DoS, etc.) |
| Deployment     | Streamlit Community Cloud |

---

## 📁 Files in This Repo

| File                          | Description                            |
|-------------------------------|----------------------------------------|
| `deploy.py`                   | Main Streamlit app                     |
| `model-CNN-LSTM.ipynb`        | File Jupyter to train the model        |
| `model_hybrid_cnn_lstm.pt`    | Trained model                          |
| `requirements.txt`            | Python dependencies for deployment     |
| `test.csv`                    | Example network flow data              |
| `predictions.csv`             | Example predictions from the test.csv  |

---

## 🧪 Sample Prediction Output

| ... | Confidence | Label  |
|-----|------------|------- |
|  0  | 0.999999   | Bot    |
|  1  | 0.989986	 | Benign |
...

---

## 🛠️ Setup Locally

```bash
git clone https://github.com/dphuocle/Intrusion_Detection_System.git
cd Intrusion_Detection_System
pip install -r requirements.txt
streamlit run deploy.py
