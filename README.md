# 🌍 ESG Oracle: ESG Forecasting with RNNs & LSTMs

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
![Models: LSTM | RNN](https://img.shields.io/badge/Models-LSTM%20%7C%20RNN-brightgreen)
![Status: Completed](https://img.shields.io/badge/Status-Completed-blueviolet)

---

## 🌱 Overview

**ESG-Oracle** is a deep learning mini-project designed to forecast **Environmental, Social, and Governance (ESG)** scores for countries around the world — specifically for the years **2025 and 2026**. ESG scores play a vital role in evaluating a country’s or organization’s sustainability practices, governance structures, and social impact — all crucial for building a more responsible and climate-conscious future.

This project combines classical time-series modeling with synthetic data generation to tackle the challenge of sparse or noisy ESG data. By leveraging both **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** models, the system forecasts country-level ESG trends. In addition, it uses a **Conditional Variational Autoencoder (CVAE)** to generate synthetic time-series data, enriching the forecasting pipeline and testing model robustness.

Whether you're exploring AI for sustainability, trying to decode ESG metrics, or simply into neural nets that care about the planet, this project shows how deep learning can support better decision-making for a greener tomorrow.

### 📦 This repository includes:
- 📈 ESG forecasting using **RNN** and **LSTM** architectures  
- 🧬 Synthetic ESG time-series generation with **CVAE**  
- 📊 Clean visualizations comparing real, predicted, and synthetic trends  
- 🧠 Modular, extensible code for future experimentation and deployments
- 
---

## 🧠 Features

- Real + synthetic ESG trend forecasting for 2025–2026  
- RNN and LSTM model comparison with evaluation metrics  
- Synthetic data from CVAE to fill in forecasting gaps  
- Clear, comparison-ready visual plots  
- Organized results, ready for extension or deployment

---

## 🛠️ Tech Stack

- Python 3.11  
- PyTorch, NumPy, Pandas  
- Scikit-learn, Matplotlib  

---

## 📊 Model Metrics

| Metric | RNN | LSTM |
|--------|-----|------|
| **MSE** | 0.01341 | 0.01348 |
| **MAE** | 0.06125 | 0.06133 |
| **R²**  | 0.94453 | 0.95118 |

*These metrics reflect the 2025–2026 forecast window using real ESG score data.*

---

## 📸 Visualizations

### ESG Trend Comparison: Real vs Synthetic vs Predicted
![ESG Trend Comparison](https://raw.githubusercontent.com/Harb1nger001/esg-oracle/main/results/visualizations/esg_trend_comparison_future.png)

---

### ESG Score Distribution Plot
![ESG Score Distribution](https://raw.githubusercontent.com/Harb1nger001/esg-oracle/main/results/visualizations/ESG_Score_distribution_plot.png)

---

### ESG Score Distribution: Real vs RNN vs LSTM
![ESG Score Distribution Real vs RNN vs LSTM](https://raw.githubusercontent.com/Harb1nger001/esg-oracle/main/results/visualizations/ESG_Score_distribution_real_rnn_lstm.png)

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Made with coffee and a little too much recursion by  
**Anamitra Majumder** ([@Harb1nger001](https://github.com/Harb1nger001))  
MSc Data Science @ IIIT Lucknow  

---

> “Because saving the planet looks better with a good R² score.” 🌍📈 

