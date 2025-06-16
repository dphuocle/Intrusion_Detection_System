import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# === Define your multi-class model ===
class CNN_LSTM_FlowClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=10, hidden_size=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, F]
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [B, T, C]
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)  # [B, num_classes]

# === Load the trained model ===
@st.cache_resource
def load_model(model_path, feature_dim, num_classes=8):
    model = CNN_LSTM_FlowClassifier(feature_dim=feature_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# === Streamlit UI ===
st.title("🛡️ Intrusion Detection System")
st.write("Upload a CSV file containing network flow data to detect different types of attacks.")

# Support optional CLI CSV path
csv_path = None
if len(sys.argv) > 1:
    csv_path = sys.argv[1]

if csv_path and os.path.isfile(csv_path):
    st.info(f"📂 Loaded file from command line: `{csv_path}`")
    df = pd.read_csv(csv_path)
else:
    uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])
    if not uploaded_file:
        st.stop()
    df = pd.read_csv(uploaded_file)

original_df = df.copy()

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# === Load and predict ===
model = load_model("model_hybrid_cnn_lstm.pt", feature_dim=X_tensor.shape[1])
with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

# === Mapping class labels ===
label_mapping = {
    0: 'Benign',
    1: 'Bot',
    2: 'DDoS_HOIC',
    3: 'DDoS_LOIC_HTTP',
    4: 'DoS_GoldenEye',
    5: 'DoS_Hulk',
    6: 'FTP_BruteForce',
    7: 'SSH_BruteForce'
}

labels = pd.Series(preds).map(label_mapping).reset_index(drop=True)
confidences = pd.Series(np.max(probs, axis=1)).reset_index(drop=True)

# === Output results ===
result_df = original_df.iloc[:len(preds)].copy().reset_index(drop=True)
result_df['Confidence'] = confidences
result_df['Label'] = labels

st.success("✅ Prediction complete!")
st.dataframe(result_df)

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")