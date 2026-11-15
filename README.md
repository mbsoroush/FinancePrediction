# 📈 Finance Prediction

> **Comparing RNN, LSTM, GRU, BiLSTM, CNN-LSTM, TCN and Transformer for Multi-Horizon Stock Forecasting (GOOG)**

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![Keras](https://img.shields.io/badge/Keras-3.3.3-red)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF)
![License](https://img.shields.io/badge/License-GPLv3-blue)

---

## 🔰 Overview
This project is a **fair, leak-free comparison of sequence models** for forecasting Google (GOOG) stock.  
It compares **7 neural architectures** against simple baselines and uses the best one to forecast the next **10 trading days**:

- 🔁 SimpleRNN  
- 🧠 LSTM  
- ⚡ GRU  
- ↔️ BiLSTM  
- 🧩 CNN-LSTM  
- 🌊 TCN (temporal convolutional network)  
- 🔭 Transformer  

Baselines: **Naive** (price stays the same), **Drift** (average trend continues) and **Ridge** regression.

The project has two main parts:
1. **Model Comparison** → Training, tuning and evaluating all models with walk-forward cross-validation and an untouched hold-out test set.  
2. **Forecasting** → Using the best model to predict the next 10 days of **price** (with an 80% uncertainty band) and **volatility**.

---

## 📦 Dataset
We used a daily Google (GOOG) stock dataset hosted on Kaggle (`google-dataset`):

- **1,258** trading days (2016-06-14 → 2021-06-11)  
- **1,239** rows after feature warm-up  
- Columns used: `date`, `open`, `high`, `low`, `close`, `volume`  

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash

git clone https://github.com/mbsoroush/FinancePrediction.git
cd FinancePrediction
pip install -r requirements.txt

```

## 🚀 Usage

To run the experiment locally:

1. ✅ Place `GOOG.csv` in the project folder (or set `CSV_PATH` in the first code cell)  
2. ✅ Run the following command:  

   ```bash
   jupyter notebook finance-prediction-portfolio.ipynb
   ```

3. ✅ Run all cells  

On Kaggle: add the dataset as input, enable a GPU, and run all cells (about **50-60 minutes** on 2x T4).

## 🧹 Preprocessing Steps

Before training, the price data went through several preprocessing steps:

1. ✅ **Feature Engineering** (7 return-based features)  
   - Daily log return  
   - Intraday range (`log(high / low)`)  
   - Open-to-close move  
   - Volume change  
   - 10-day rolling volatility  
   - Distance from the 20-day moving average  
   - RSI (14)  

2. ✅ **Windowing**  
   - Input: the last **60 trading days** of features  
   - Target: the cumulative log return over the next **10 days**, predicted directly in one forward pass  
   - Predictions are converted back to dollars  

3. ✅ **Scaling (no leakage)**  
   - Scalers and target statistics are fit on **training rows only**, separately for every fold  

4. ✅ **Dataset Splitting**  
   - Hold-out test set → last **15%** of the timeline (176 samples), never used for training, tuning or selection  
   - **4 expanding-window folds** on the remaining 85% (validation blocks of about 6 months each)  
   - Each sample's 10-day target stays inside its own split  

---

📌 After preprocessing, the data was ready for training on every model.

## 🧠 Model Training Steps

All models were trained on Kaggle with GPU acceleration.  
Below are the main steps:

1. ✅ **Model Selection**  
   - 7 neural architectures + 3 baselines  
   - All share the same input `(60 days x 7 features)` and the same output head  

2. ✅ **Training Environment**  
   - Kaggle GPU (2x Tesla T4)  
   - TensorFlow 2.16.1, Keras 3.3.3  

3. ✅ **Training Setup**  
   - Optimizer: **AdamW** with gradient clipping  
   - Loss: MSE on standardized targets  
   - Batch size: **64**, up to **100** epochs  
   - Early stopping (patience 12, best weights restored) and learning-rate halving on plateau  

4. ✅ **Hyperparameter Search**  
   - Equal budget for every architecture: its default configuration + **5 random configurations**  
   - Scored on two tuning folds with one seed  
   - First and last folds are never used for tuning  

5. ✅ **Model Selection Criterion**  
   - Final CV with **3 seeds** on all 4 folds  
   - Metric: mean RMSE **relative to the naive forecast** (scale-free, below 1 beats naive)  

6. ✅ **Evaluation Metrics**  
   - RMSE, MAE, MAPE, R², directional accuracy  
   - **Diebold-Mariano tests** (HAC variance) against Naive and Drift  

---

📌 After selection, the best architecture was refit on the most recent data for the final forecasts.

## 📊 Results

Walk-forward CV (mean RMSE relative to naive, lower is better) and hold-out test (Sep 2020 → Jun 2021):

| Model | CV rel. RMSE | Test RMSE ($, mean over seeds) | DM p-value vs Naive (seed ensemble) |
|---|---|---|---|
| Transformer | **0.957** | 75.50 | 0.004 |
| BiLSTM | 0.970 | 75.85 | 0.006 |
| Drift (baseline) | 0.998 | 75.54 | 0.005 |
| Ridge (baseline) | 0.998 | 75.61 | 0.006 |
| Naive (baseline) | 1.000 | 77.89 | - |
| TCN | 1.001 | 77.01 | 0.054 |
| CNN-LSTM | 1.002 | 75.76 | 0.003 |
| GRU | 1.003 | 76.37 | 0.006 |
| LSTM | 1.011 | 78.09 | 0.942 |
| SimpleRNN | 1.013 | 78.27 | 0.293 |

**Key findings**

1. ✅ The **Transformer** was selected and beats naive on the test set (RMSE **$75.08** as a 3-seed ensemble vs $77.89, p = 0.004)  
2. ✅ It is **statistically indistinguishable from the Drift baseline** (p = 0.46): most of the gain reflects the uptrend of the test period  
3. ✅ The advantage of the best models is **regime-dependent**: about 10% better than naive in two folds, worse than naive in another  
4. ✅ Validation loss is lowest within the first few epochs for almost every model, which points to a **weak predictable signal** in daily OHLCV features  
5. ✅ The **80% prediction interval** covered **78.1%** of test outcomes (well calibrated)  
6. ✅ The **volatility** model beats a persistence baseline but not a constant long-run mean (RMSE 0.00748 vs 0.00696)  

---

📌 For daily GOOG data, sequence models add at most a small, period-dependent edge over simple trend baselines.

## 🔮 Example Forecast Steps

Here's what the final forecast looks like:

1. ✅ **Input**  
   - The last 60 trading days of features up to **2021-06-11** (close: `2513.93`)  

2. ✅ **Model Processing**  
   - The Transformer is refit on the most recent data (3-seed ensemble)  
   - One forward pass produces all 10 horizons  

3. ✅ **Prediction Output**  
   - 📅 **Day 1** (2021-06-14) → `2506.98` (80% band: `2466.98 – 2562.32`)  
   - 📅 **Day 10** (2021-06-25) → `2445.08` (80% band: `2364.43 – 2627.13`)  
   - 📉 Volatility → about **1.47% per day** (~23% annualised)  

4. ✅ **Visualization**  
   - A chart with the last 90 days, the forecast path and the 80% band  

---

📌 The last close lies inside the band at every horizon, so the forecast is a central estimate with uncertainty, not a trading signal.

## 🧪 Troubleshooting

Common issues and their solutions when running the project:

1. ❌ **`FileNotFoundError` for `GOOG.csv`**  
   - Cause: The dataset path does not exist  
   - ✅ Solution:  
     - Place `GOOG.csv` next to the notebook, or set `CSV_PATH` in the first code cell  

2. ❌ **Version or Keras errors**  
   - Cause: A different TensorFlow/Keras version than the one used in the experiment  
   - ✅ Solution:  
     - Install the pinned version:  
       ```bash
       pip install -r requirements.txt
       ```  

3. ❌ **Results differ slightly from the README**  
   - Cause: GPU operations are not fully deterministic, even with fixed seeds  
   - ✅ Solution:  
     - Small variations are expected; conclusions should stay the same  

4. ❌ **Run takes too long**  
   - Cause: Hyperparameter search + 4 folds x 3 seeds x 7 models  
   - ✅ Solution:  
     - Reduce `N_TRIALS` and `SEEDS` in the configuration cell, or run on a GPU  

5. ❌ **CUDA / XLA warnings in the logs**  
   - Cause: Normal TensorFlow startup messages  
   - ✅ Solution:  
     - They are harmless and can be ignored  

---

📌 Following these steps should resolve most issues encountered when running the notebook.

## 📦 Deliverables

The project includes the following components:

1. ✅ Experiment notebook → `finance-prediction-portfolio.ipynb`  
2. ✅ Dependencies → `requirements.txt`  
3. ✅ Generated results → `results/` (CV summary, test table, Diebold-Mariano results, best hyperparameters, price forecast)  
4. ✅ Dataset → Google (GOOG) daily stock data on Kaggle (`google-dataset`)  

---

## 🏁 Conclusion

1. ✅ **Objective Achieved**  
   - Compared 7 sequence models and 3 baselines with a rigorous, leak-free protocol  

2. ✅ **Key Features**  
   - Walk-forward cross-validation + untouched hold-out test  
   - Equal tuning budget and 3 seeds per model  
   - Statistical significance tests and calibrated prediction intervals  
   - 10-day price and volatility forecasts  

3. ✅ **Impact**  
   - Shows that a fair evaluation matters: a Transformer or BiLSTM can beat naive, but none of the neural models demonstrably beats a simple drift forecast  

4. ✅ **Limitations**  
   - One stock and about 1,200 daily observations  
   - Validation folds and the test set cover different market regimes  
   - Small random hyperparameter budget  
   - No transaction costs or trading strategy; forecasts are for research only, not financial advice  

5. ✅ **Future Improvements**  
   - More tickers and longer histories  
   - A stronger volatility baseline (HAR-style) and level correction  
   - Cost-aware backtesting and probabilistic models  

---

📌 With this project, we showed that **careful evaluation** matters more than model complexity when forecasting noisy financial time series.


## ⚖️ License

This project is licensed under the [GNU General Public License v3.0](https://github.com/mbsoroush/FinancePrediction/blob/main/LICENSE).  
You are free to use, modify and distribute the code under the terms of the GPL-3.0; derivative works must be released under the same license.

The dataset is not covered by this license: it remains subject to the terms of its original source on Kaggle.
