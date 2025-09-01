
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from ta.momentum import RSIIndicator
import qrcode
from PIL import Image


def run_analysis():
    # 1. Data Collection
    stock_symbol = "^NSEI"  # NIFTY 50 index
    data = yf.download(stock_symbol, start="2020-05-22", end="2025-05-22")
    df = data[['Close']].copy()

    # 2. Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 3. Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1, verbose=0)  # Reduce epochs for speed

    # 4. Evaluate LSTM Model
    loss = model.evaluate(X_test, y_test, verbose=0)

    # 5. Predict with LSTM
    predicted_prices = model.predict(X_test, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 6. Calculate LSTM Metrics
    mse_lstm = float(mean_squared_error(y_test_unscaled, predicted_prices))
    rmse_lstm = float(np.sqrt(mse_lstm))

    # 7. Train Random Forest Model
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X_train_rf, y_train)
    rf_predictions = rf_model.predict(X_test_rf)

    # 8. Evaluate Random Forest Model
    mse_rf = float(mean_squared_error(y_test, rf_predictions))

    # 9. Train XGBoost Model
    xgb_model = XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train_rf, y_train)
    xgb_predictions = xgb_model.predict(X_test_rf)

    # 10. Evaluate XGBoost
    mse_xgb = float(mean_squared_error(y_test, xgb_predictions))
    rmse_xgb = float(np.sqrt(mse_xgb))

    return {
        "lstm_mse": mse_lstm,
        "lstm_rmse": rmse_lstm,
        "rf_mse": mse_rf,
        "xgb_mse": mse_xgb,
        "xgb_rmse": rmse_xgb
    }

# Financial Metrics Calculator
def calculate_eps(net_income, preferred_dividends, shares_outstanding):
    """Calculate Earnings Per Share (EPS)."""
    try:
        if shares_outstanding == 0:
            return "Error: Shares Outstanding cannot be zero."
        eps = (net_income - preferred_dividends) / shares_outstanding
        return round(eps, 2)
    except Exception as e:
        return f"Error calculating EPS: {e}"

def calculate_pe_ratio(stock_price, eps):
    """Calculate Price-to-Earnings (P/E) Ratio."""
    try:
        if isinstance(eps, str):  # If EPS is error message
            return "Error: Cannot calculate P/E Ratio without valid EPS."
        if eps == 0:
            return "Error: EPS cannot be zero."
        pe_ratio = stock_price / eps
        return round(pe_ratio, 2)
    except Exception as e:
        return f"Error calculating P/E Ratio: {e}"

def calculate_roe(net_income, shareholders_equity):
    """Calculate Return on Equity (ROE)."""
    try:
        if shareholders_equity == 0:
            return "Error: Shareholders' Equity cannot be zero."
        roe = net_income / shareholders_equity
        return round(roe * 100, 2)  # return percentage
    except Exception as e:
        return f"Error calculating ROE: {e}"

def main():
    print("\n Financial Metrics Calculator")
    print("---------------------------------")
    # Input financial data
    net_income = float(input("Enter Net Income ($): "))
    shareholders_equity = float(input("Enter Shareholders' Equity ($): "))
    preferred_dividends = float(input("Enter Preferred Dividends ($, enter 0 if none): "))
    shares_outstanding = float(input("Enter Weighted Average Shares Outstanding: "))
    stock_price = float(input("Enter Current Stock Price ($): "))

    # Calculate metrics
    eps = calculate_eps(net_income, preferred_dividends, shares_outstanding)
    pe_ratio = calculate_pe_ratio(stock_price, eps)
    roe = calculate_roe(net_income, shareholders_equity)

    # Print results
    print("\nResults:")
    print(f"Earnings Per Share (EPS): {eps}")
    print(f"Price-to-Earnings (P/E) Ratio: {pe_ratio}")
    print(f"Return on Equity (ROE): {roe}%")

if __name__ == "__main__":
    main()

# Function to generate QR code
def generate_qr_code(url, filename="qr_code.png"):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image.save(filename)
    return qr_image

# 9. Visualize Results
# LSTM Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Price')
plt.plot(predicted_prices, label='LSTM Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Random Forest Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Price', color='blue', linestyle='--')
plt.plot(scaler.inverse_transform(rf_predictions.reshape(-1, 1)), 
         label='Random Forest Predicted Price', color='red')
plt.title('Random Forest Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# XGBoost Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Price', color='blue', linestyle='--')
plt.plot(scaler.inverse_transform(xgb_predictions.reshape(-1, 1)), 
         label='XGBoost Predicted Price', color='orange')
plt.title('XGBoost Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Combined Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Price', color='blue', linestyle='--')
plt.plot(predicted_prices, label='LSTM Predicted Price', color='green')
plt.plot(scaler.inverse_transform(rf_predictions.reshape(-1, 1)), 
         label='Random Forest Predicted Price', color='red')
plt.title('Stock Price Prediction: LSTM vs Random Forest')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# Combined Visualization with QR Code
plt.figure(figsize=(15, 15))  # Increased size to accommodate QR code
plt.plot(y_test_unscaled, label='Actual Price', color='blue', linestyle='--')
plt.plot(predicted_prices, label='LSTM Predicted Price', color='green')
plt.plot(scaler.inverse_transform(rf_predictions.reshape(-1, 1)), 
         label='Random Forest Predicted Price', color='red')
plt.plot(scaler.inverse_transform(xgb_predictions.reshape(-1, 1)), 
         label='XGBoost Predicted Price', color='orange')
plt.title('Stock Price Prediction: LSTM vs Random Forest vs XGBoost')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
# Generate QR code separately
qr_url = "https://finance.yahoo.com/quote/%5ENSEI/history/"  
qr_image = generate_qr_code(qr_url, "nifty50_qr_code.png")

# Optional: Show QR code separately (using PIL)
qr_image.show()  # Opens in default image viewer

# Optional: Display QR code in a matplotlib window separately
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

qr_img = Image.open("nifty50_qr_code.png")
qr_img_np = np.array(qr_img)

plt.figure(figsize=(3, 3))
plt.imshow(qr_img_np)
plt.axis('off')
plt.title("NIFTY 50 QR Code")
plt.show()
