# 🌍 ESG Oracle: ESG Forecasting with RNNs & LSTMs

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
![Models: LSTM | RNN](https://img.shields.io/badge/Models-LSTM%20%7C%20RNN-brightgreen)
![Status: Completed](https://img.shields.io/badge/Status-Completed-blueviolet)

---

## 🌱 Overview

**ESG-Oracle** is a deep learning mini-project focused on forecasting **ESG (Environmental, Social, and Governance)** scores across countries for 2025–2026. It blends classical deep learning with synthetic data generation to explore real-world sustainability trends.

This repo features:
- 📈 ESG forecasting using **RNN** and **LSTM**
- 🧬 Synthetic data generation via **CVAE**
- 📊 Clean and detailed visualizations
- 🧠 Scalable, modular codebase

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
- TorchQuantum, Qiskit, PennyLane (Quantum ready)

---

## 📁 Project Structure

```
esg-oracle/
├── results/
│   ├── metrics/
│   │   ├── lstm_metrics.csv
│   │   └── rnn_metrics.csv
│   └── predictions/
│       ├── esg_forecasts_2025_2026.csv
│       └── esg_forecasts_lstm_2025_2026.csv
├── visualizations/
│   ├── forecast_plotter.py
│   ├── tsne_plotter.py
│   ├── trend_metrics_plot.py
│   └── model_comparison_plot.py
```

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
