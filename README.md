# sp500-lstm-forecasting
S&amp;P 500 stock price prediction using LSTM neural networks with yfinance data. Features model training, evaluation metrics (MAE/RMSE/R²), visualizations, and 30-day future forecasts. Built with TensorFlow/Keras.


# S&P 500 LSTM Price Prediction & Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-brightgreen)](https://keras.io/)

Advanced **Long Short-Term Memory (LSTM)** model for predicting S&P 500 (^GSPC) closing prices using historical data from yfinance. Achieves strong R² scores across train/val/test sets with 30-day future forecasting.

## 🚀 Features
- Fetches real-time S&P 500 data (2024–2030) via `yfinance`.
- 3-layer LSTM architecture (100→100→50 units) with Dropout (0.2) and EarlyStopping.
- 80/10/10 train/val/test split; sequence length=100.
- Comprehensive metrics: MAE, RMSE, R², directional accuracy.
- Visualizations: loss curves, predictions vs actual, error histograms, 30-day forecasts.
- Predicts future prices (e.g., ~X% change in 30 days—run to see latest).

## 📊 Results (Example)
| Set       | MAE    | RMSE   | R²     |
|-----------|--------|--------|--------|
| Training | $X.XX | $X.XX | 0.XXXX |
| Validation | $X.XX | $X.XX | 0.XXXX |
| Test     | $X.XX | $X.XX | 0.XXXX |

*Replace with your model's outputs. Generated plots saved as `sp500_lstm_prediction.png` & `sp500_future_forecast.png`.*

![Prediction Plot](sp500_lstm_prediction.png)
![Forecast Plot](sp500_future_forecast.png)

## 📦 Installation
1. Clone repo: `git clone https://github.com/YOUR_USERNAME/sp500-lstm-forecasting.git`
2. Install deps: `pip install yfinance tensorflow scikit-learn pandas numpy matplotlib`
3. Run: `python lstm_sp500_predictor.py`

## 📁 Structure

sp500-lstm-forecasting/
├── lstm_sp500_predictor.py # Main script (your code)
├── sp500_lstm_prediction.png
├── sp500_future_forecast.png
├── requirements.txt
└── README.md


## 🔮 Usage
Run the script for instant training, evaluation, and 30-day forecast. Customize `seq_length`, `days`, or dates.

```python
# Example future prediction output:
# Predicted price after 30 days: $XXXX.XX
# Current price: $XXXX.XX
# Predicted change: X.XX%
