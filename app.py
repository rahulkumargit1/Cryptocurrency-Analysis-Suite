import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from tensorflow import keras
from flask import Flask, render_template, request, send_file
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import requests
import time
import re
from pathlib import Path

# Set plot style
plt.style.use("fivethirtyeight")

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Ensure static directory exists
Path("static").mkdir(exist_ok=True)

# CryptoCompare configuration
CRYPTOCOMPARE_BASE_URL = 'https://min-api.cryptocompare.com/data'

# Fetch list of all cryptocurrencies from CryptoCompare
def get_crypto_list():
    url = f'{CRYPTOCOMPARE_BASE_URL}/all/coinlist'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['Response'] != 'Success':
            print(f"CryptoCompare coinlist error: {data.get('Message', 'Unknown error')}")
            return {}
        return data['Data']
    except requests.RequestException as e:
        print(f"Error fetching coin list: {e}")
        return {}

# Cache the coin list for performance
crypto_list = get_crypto_list()

def get_historical_data(symbol, start, end, max_retries=5):
    end_timestamp = int(end.timestamp())
    fsym, tsym = symbol.split('/')
    url = f'{CRYPTOCOMPARE_BASE_URL}/histoday'
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'toTs': end_timestamp,
        'limit': 2000
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['Response'] != 'Success':
                print(f"CryptoCompare error: {data.get('Message', 'Unknown error')}")
                return []
            prices = data['Data']
            return [(item['time'] * 1000, item['close']) for item in prices if item['close'] > 0]
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(2 ** attempt)
    return []

def get_current_price(symbol):
    fsym, tsym = symbol.split('/')
    url = f'{CRYPTOCOMPARE_BASE_URL}/price'
    params = {
        'fsym': fsym,
        'tsyms': tsym
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return str(data[tsym])
    except requests.RequestException as e:
        print(f"Error fetching current price: {e}")
        return None

# Create a simple model if the trained model doesn't exist
def create_simple_model():
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(100, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse')
    return model

# Load or create the model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'stock_dl_model.keras')
if os.path.exists(model_path):
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='rmsprop', loss='mse')
    except Exception as e:
        print(f"Error loading model: {e}, creating new model")
        model = create_simple_model()
else:
    print("Model file not found, creating new model")
    Path("models").mkdir(exist_ok=True)
    model = create_simple_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    searched_ticker = None
    stock_details = None
    plot_path_ema_20_50 = None
    plot_path_ema_100_200 = None
    plot_path_prediction = None
    plot_path_future_forecast = None
    dataset_link = None
    tradingview_symbol = None
    error_message = None

    if request.method == 'POST':
        stock = request.form.get('stock', 'BTC-USD').upper()
        searched_ticker = stock

        if not re.match(r'^[A-Z]+-USD$', searched_ticker):
            error_message = "Error: Invalid ticker format. Use format like 'COIN-USD' (e.g., BTC-USD, ETH-USD)."
            return render_template('index.html', error_message=error_message)

        symbol = searched_ticker.replace('-', '/')
        fsym = symbol.split('/')[0]

        if fsym not in crypto_list:
            error_message = f"Error: Cryptocurrency {fsym} not found. Try a valid ticker like BTC-USD, ETH-USD, or LTC-USD."
            return render_template('index.html', error_message=error_message)

        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        try:
            historical_data = get_historical_data(symbol, start, end)
            if not historical_data:
                error_message = f"Error: No historical data available for {searched_ticker}."
                return render_template('index.html', error_message=error_message)

            dates = [dt.datetime.utcfromtimestamp(data[0] / 1000) for data in historical_data]
            close_prices = [float(data[1]) for data in historical_data]
            df = pd.DataFrame({'Close': close_prices}, index=dates)

            current_price = get_current_price(symbol)
            if not current_price:
                error_message = f"Error: Unable to fetch current price for {searched_ticker}."
                return render_template('index.html', error_message=error_message)

            coin_data = crypto_list.get(fsym, {})
            crypto_desc = coin_data.get('Description', f"{fsym} is a cryptocurrency traded against USD.")
            stock_details = {
                "name": searched_ticker,
                "current_price": current_price,
                "description": f"{crypto_desc} Data sourced from CryptoCompare."
            }
        except Exception as e:
            print(f"Error fetching data from CryptoCompare: {e}")
            error_message = f"Error: Unable to fetch data for {searched_ticker}. Please try again later."
            return render_template('index.html', error_message=error_message)

        tradingview_symbol = f"CRYPTOCOMPARE:{searched_ticker.replace('-', '')}"

        earliest_date = df.index.min()
        if earliest_date > start:
            print(f"⚠️ Data starts from {earliest_date.date()}. Adjusting range.")
            start = earliest_date

        full_date_range = pd.date_range(start=start, end=end, freq='D')
        df = df.reindex(full_date_range, method='ffill')

        # Calculate EMAs
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

        # Prepare data for AI prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        x_data = [scaled_data[i-100:i] for i in range(100, len(scaled_data))]
        x_data = np.array(x_data)
        if x_data.size == 0:
            error_message = "Error: Insufficient data for prediction (need at least 100 days)."
            return render_template('index.html', error_message=error_message)
        y_predicted_scaled = model.predict(x_data, verbose=0)
        y_predicted = scaler.inverse_transform(y_predicted_scaled).flatten()
        y_data = scaler.inverse_transform(scaled_data[100:]).flatten()

        # 30-Day Future Forecast
        future_days = 30
        future_predictions = []
        future_dates = pd.date_range(start=end + dt.timedelta(days=1), periods=future_days, freq='D')
        last_sequence = scaled_data[-100:].copy()
        for _ in range(future_days):
            pred = model.predict(last_sequence.reshape(1, 100, 1), verbose=0)
            future_predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        confidence_lower = future_predictions * 0.9
        confidence_upper = future_predictions * 1.1

        # Function to save plots
        def save_plot(fig, filename):
            path = os.path.join("static", filename)
            fig.savefig(path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            return filename

        # Short-term plot (last 365 days)
        short_term_start = end - dt.timedelta(days=365)
        df_recent = df.loc[df.index >= short_term_start]
        ema20_recent = ema20.loc[df.index >= short_term_start]
        ema50_recent = ema50.loc[df.index >= short_term_start]
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df_recent['Close'], 'y', label='Closing Price')
        ax1.plot(ema20_recent, 'g', label='EMA 20')
        ax1.plot(ema50_recent, 'r', label='EMA 50')
        ax1.set_title("Short-Term (Last 365 Days, 20 & 50 Days EMA)")
        ax1.legend()
        plot_path_ema_20_50 = save_plot(fig1, f"ema_20_50_{searched_ticker}.png")

        # Long-term plot (full dataset)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df['Close'], 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title(f"Long-Term ({start.date()} to {end.date()}, 100 & 200 Days EMA)")
        ax2.legend()
        plot_path_ema_100_200 = save_plot(fig2, f"ema_100_200_{searched_ticker}.png")

        # AI prediction plot
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(df.index[100:], y_data, 'g', label="Original Price")
        ax3.plot(df.index[100:], y_predicted, 'r', label="Predicted Price")
        ax3.set_title("AI Price Prediction")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Price (USD)")
        ax3.legend()
        plot_path_prediction = save_plot(fig3, f"stock_prediction_{searched_ticker}.png")

        # 30-Day Future Forecast Plot
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(df.index[-100:], df['Close'].values[-100:], 'g', label="Historical Price")
        ax4.plot(future_dates, future_predictions, 'r', label="Forecasted Price")
        ax4.fill_between(future_dates, confidence_lower, confidence_upper, color='r', alpha=0.1, label="Confidence Interval")
        ax4.set_title(f"30-Day Price Forecast for {searched_ticker}")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Price (USD)")
        ax4.legend()
        plot_path_future_forecast = save_plot(fig4, f"future_forecast_{searched_ticker}.png")

        # Save dataset as CSV
        csv_path = f"static/{searched_ticker}_dataset.csv"
        df.to_csv(csv_path)
        dataset_link = csv_path

        return render_template('index.html',
                               searched_ticker=searched_ticker,
                               stock_details=stock_details,
                               plot_path_ema_20_50=plot_path_ema_20_50,
                               plot_path_ema_100_200=plot_path_ema_100_200,
                               plot_path_prediction=plot_path_prediction,
                               plot_path_future_forecast=plot_path_future_forecast,
                               dataset_link=dataset_link,
                               tradingview_symbol=tradingview_symbol,
                               error_message=error_message)

    return render_template('index.html', error_message=error_message)

@app.route('/download/<filename>')
def download_file(filename):
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-.]', '', filename)
    file_path = os.path.join("static", safe_filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join('static', 'images', 'favicon.png'), mimetype='image/png')

if __name__ == '__main__':
    # For local development only; use Gunicorn or similar for production
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
