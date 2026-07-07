# ==========================================================
# IMPORTS
# ==========================================================

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================================
# YOUR MODEL (UNCHANGED)
# ==========================================================

class ECG_PPG_HRV_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.ecg_net = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.ppg_net = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.hrv_net = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 16 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, ecg, ppg, hrv):
        ecg = ecg.unsqueeze(1)
        ppg = ppg.unsqueeze(1)

        ecg_feat = self.ecg_net(ecg).squeeze(-1)
        ppg_feat = self.ppg_net(ppg).squeeze(-1)
        hrv_feat = self.hrv_net(hrv)

        x = torch.cat([ecg_feat, ppg_feat, hrv_feat], dim=1)
        return self.classifier(x).squeeze(1)


# ==========================================================
# LOAD MODEL (ASSUME TRAINED)
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECG_PPG_HRV_Model().to(device)

# Optional: load trained weights
# model.load_state_dict(torch.load("model.pth", map_location=device))

model.eval()


# ==========================================================
# SAMPLE INPUT (REPLACE WITH REAL DATA)
# ==========================================================

ecg_sample = torch.randn(1, 5120).to(device)
ppg_sample = torch.randn(1, 5120).to(device)

hrv_sample = torch.tensor([[70, 75, 40, 30, 10, 0.2, 60, 90, 0.1, 0.5, 0.3]],
                          dtype=torch.float32).to(device)


# ==========================================================
# PREDICTION
# ==========================================================

with torch.no_grad():
    output = model(ecg_sample, ppg_sample, hrv_sample)
    prob = torch.sigmoid(output).item()

state = "Drowsy" if prob > 0.5 else "Alert"

print("\nPrediction:", state)


# ==========================================================
# HRV INTERPRETATION
# ==========================================================

hrv_vals = hrv_sample.cpu().numpy()[0]

rr_mean = hrv_vals[0]
hr = hrv_vals[1]
sdnn = hrv_vals[2]
rmssd = hrv_vals[3]

if sdnn < 50:
    hrv_desc = "low heart rate variability"
    physio = "reduced autonomic nervous system activity and fatigue"
elif sdnn < 100:
    hrv_desc = "moderate heart rate variability"
    physio = "normal autonomic balance"
else:
    hrv_desc = "high heart rate variability"
    physio = "active autonomic nervous system"


# ==========================================================
# LOAD ONLINE TINYLLAMA
# ==========================================================

print("\nLoading TinyLlama (online)...")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)


# ==========================================================
# PROMPT (HIGH QUALITY)
# ==========================================================

prompt = f"""
You are a biomedical expert.

Driver state: {state}
ECG shows {hrv_desc}.
Heart rate is {hr:.2f} bpm and SDNN is {sdnn:.2f}.

Write a clear explanation in exactly 4 sentences.
"""


# ==========================================================
# GENERATE EXPLANATION
# ==========================================================

inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

outputs = llm.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove prompt
final_output = result.replace(prompt, "").strip()


# ==========================================================
# OUTPUT
# ==========================================================

print("\n========== SLM EXPLANATION ==========\n")
print(final_output)