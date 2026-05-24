# 🔧 Predictive Maintenance & Intelligent RUL Forecasting Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

---

## 📋 Overview

An end-to-end **Predictive Maintenance** system that leverages the NASA C-MAPSS Turbofan Engine Degradation dataset to:

- **Detect anomalies** in sensor readings using statistical and ML-based methods
- **Predict Remaining Useful Life (RUL)** using deep learning (LSTM) and classical ML models
- **Visualize insights** through an interactive Streamlit dashboard with Plotly charts

> Built with a clean, modular, enterprise-grade architecture suitable for production deployment.

---

## 🏗️ Project Structure

```
Predictive Maintenance System/
│
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup (optional pip install)
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment variable template
├── main.py                         # Application entry point
│
├── configs/                        # Configuration files
│   ├── __init__.py
│   ├── settings.py                 # Global settings & hyperparameters
│   └── logging_config.py           # Logging configuration
│
├── data/                           # Dataset storage
│   ├── nasa/                       # NASA C-MAPSS dataset
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/                  # Preprocessed / feature-engineered data
│
├── src/                            # Core source code
│   ├── __init__.py
│   ├── preprocessing/              # Data loading & feature engineering
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Load & parse raw C-MAPSS data
│   │   ├── feature_engineering.py  # Sensor normalization, rolling stats, RUL labels
│   │   └── data_pipeline.py        # End-to-end preprocessing pipeline
│   │
│   ├── anomaly_detection/          # Anomaly / health monitoring
│   │   ├── __init__.py
│   │   ├── statistical.py          # Z-score, IQR-based anomaly detection
│   │   ├── ml_detector.py          # Isolation Forest / One-Class SVM
│   │   └── detector_pipeline.py    # Anomaly detection orchestrator
│   │
│   ├── rul_prediction/             # Remaining Useful Life models
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # LSTM / BiLSTM deep learning model
│   │   ├── classical_models.py     # Random Forest, SVR, Gradient Boosting
│   │   ├── model_trainer.py        # Training loop & callbacks
│   │   └── model_evaluator.py      # Evaluation metrics (RMSE, MAE, R², scoring)
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── logger.py               # Custom logger setup
│       ├── helpers.py              # Misc helper functions
│       └── validators.py           # Input validation utilities
│
├── dashboard/                      # Streamlit dashboard
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit app
│   ├── pages/                      # Multi-page dashboard
│   │   ├── __init__.py
│   │   ├── overview.py             # Fleet health overview
│   │   ├── anomaly_view.py         # Anomaly detection visualizations
│   │   ├── rul_view.py             # RUL prediction results
│   │   └── model_comparison.py     # Model performance comparison
│   ├── components/                 # Reusable UI components
│   │   ├── __init__.py
│   │   ├── charts.py               # Plotly chart builders
│   │   ├── sidebar.py              # Sidebar navigation & filters
│   │   └── metrics_cards.py        # KPI metric cards
│   └── assets/                     # Static assets (CSS, images)
│       └── style.css               # Custom Streamlit theme
│
├── models/                         # Saved / serialized models
│   └── .gitkeep
│
├── outputs/                        # Logs, reports, exported results
│   ├── logs/
│   │   └── .gitkeep
│   ├── reports/
│   │   └── .gitkeep
│   └── figures/
│       └── .gitkeep
│
├── notebooks/                      # Jupyter notebooks for EDA & prototyping
│   └── 01_exploratory_analysis.ipynb
│
└── tests/                          # Unit & integration tests
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_feature_engineering.py
    └── test_models.py
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/predictive-maintenance-system.git
cd predictive-maintenance-system
```

### 2. Create a Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (optional)
```

### 5. Run the Pipeline

```bash
# Run preprocessing + training
python main.py

# Launch the dashboard
streamlit run dashboard/app.py
```

---

## 📊 Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation** dataset:

| File              | Description                                      |
|-------------------|--------------------------------------------------|
| `train_FD001.txt` | Training data — run-to-failure engine trajectories |
| `test_FD001.txt`  | Test data — partial engine trajectories           |
| `RUL_FD001.txt`   | Ground truth RUL values for test engines          |

**Features:** 3 operational settings + 21 sensor measurements per engine cycle.

---

## 🧠 Models

| Model              | Type         | Purpose                          |
|---------------------|-------------|----------------------------------|
| LSTM / BiLSTM       | Deep Learning | Sequence-based RUL prediction    |
| Random Forest       | Classical ML  | Baseline RUL regression          |
| Gradient Boosting   | Classical ML  | Ensemble RUL regression          |
| Isolation Forest    | Anomaly Det.  | Unsupervised anomaly scoring     |
| One-Class SVM       | Anomaly Det.  | Boundary-based anomaly detection |

---

## 🛠️ Tech Stack

| Technology      | Purpose                        |
|-----------------|--------------------------------|
| **Python 3.9+** | Core language                  |
| **Streamlit**   | Interactive dashboard          |
| **Plotly**      | Rich, interactive charts       |
| **TensorFlow**  | Deep learning (LSTM models)    |
| **Scikit-learn**| Classical ML & preprocessing   |
| **Pandas**      | Data manipulation              |
| **NumPy**       | Numerical computation          |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  Built with ❤️ for Predictive Maintenance Research
</p>
