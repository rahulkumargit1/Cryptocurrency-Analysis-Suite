# Cryptocurrency Analysis Suite
<div align="center">
  <img src="https://github.com/rahulkumargit1/Cryptocurrency-Analysis-Suite/blob/main/CRYPTO.JPG" alt="Rahul Kumar's GitHub Cover" style="width: 100%; max-width: 800px; height: auto; border-radius: 10px;">
</div>

A Flask-based web application for advanced cryptocurrency analysis, leveraging machine learning for price predictions and technical indicators for market insights.

## Overview

The Cryptocurrency Analysis Suite is a powerful platform designed to provide real-time cryptocurrency data, technical analysis, and AI-driven price predictions. Built with Flask, TensorFlow, and CryptoCompare API, it offers users an interactive interface to analyze cryptocurrencies using historical data, Exponential Moving Averages (EMAs), and a custom deep learning model for forecasting.

## Features

- **Real-Time Market Data**: Fetch current prices and historical data using the CryptoCompare API.
- **Technical Analysis**:
  - 20 & 50-day EMAs for short-term trends.
  - 100 & 200-day EMAs for long-term market analysis.
- **AI-Powered Predictions**:
  - Machine learning model for historical price predictions.
  - 30-day price forecast with confidence intervals.
- **Interactive Charts**:
  - Live TradingView chart for real-time price tracking.
  - Matplotlib-generated plots for EMAs and predictions.
- **Data Export**: Download historical datasets as CSV files.
- **User-Friendly Interface**: Input cryptocurrency tickers (e.g., BTC-USD) to generate comprehensive analysis.

## Prerequisites

- Python 3.8+
- Flask
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Requests
- Scikit-learn

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/cryptocurrency-analysis-suite.git
   cd cryptocurrency-analysis-suite
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source(literally type source) venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install flask tensorflow pandas numpy matplotlib requests scikit-learn
   ```

4. **Prepare the Deep Learning Model**:
   - Ensure the pre-trained model (`stock_dl_model.keras`) is placed in the project root or update the path in `app.py`:
     ```python
     model_path = 'path/to/your/stock_dl_model.keras'
     ```
   - Note: The model must be trained or downloaded separately, as it’s not included in the repository.

5. **Create a `static` Folder**:
   - Ensure a `static` folder exists in the project root to store generated plots and the favicon (`static/images/favicon.png`).

## Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will start on `http://localhost:5000`.

2. **Analyze a Cryptocurrency**:
   - Open the app in a browser.
   - Enter a valid ticker (e.g., `BTC-USD`, `ETH-USD`) in the input field and click "Analyze".
   - View real-time charts, EMA plots, AI predictions, and download historical data.

3. **Supported Tickers**:
   - Use Yahoo Finance-style tickers (e.g., `BTC-USD`, `LTC-USD`).
   - The app validates tickers against CryptoCompare’s coin list.

## Project Structure

- `app.py`: Main Flask application handling routes, data fetching, and plot generation.
- `index.html`: Front-end template for the web interface.
- `static/`: Directory for storing generated plots, CSV files, and favicon.
- `stock_dl_model.keras`: Pre-trained deep learning model (not included; must be provided).

## Team Members

- **Rahul Kumar** (USN: 4SH21CS104)
- **Prakruthi Shetty** (USN: 4SH21CS102)
- **Nidhisha Shetty** (USN: 4SH21CS094)
- **Arpan Shetty** (USN: 4SH21CS023)

## Notes

- **API Limitations**: The CryptoCompare API has rate limits. Ensure proper error handling for high-frequency requests.
- **Model Dependency**: The app requires a pre-trained Keras model. Train one using historical crypto data or obtain a compatible model.
- **Data Availability**: Some cryptocurrencies may have limited historical data, affecting EMA calculations and predictions.
- **Static Assets**: Ensure the `static` folder is writable for plot and CSV generation.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements.

## License

© 2025 Advanced Cryptocurrency Analysis Platform. All rights reserved.
