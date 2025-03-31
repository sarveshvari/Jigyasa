# Jigyasafrom flask import Flask, jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

app = Flask(__name__)

# ✅ Live Data Fetch karne ka Function (Yahoo Finance ya Binance API)
def fetch_live_data():
    url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=50"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'quote_asset_volume', 'num_trades', 
                                     'taker_buy_base', 'taker_buy_quote', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    
    # ✅ Indicators Calculate Karein
    df['ema_4'] = df['close'].ewm(span=4, adjust=False).mean()
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_16'] = df['close'].ewm(span=16, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    df['rsi_5'] = 100 - (100 / (1 + df['close'].diff().rolling(5).mean() / df['close'].diff().rolling(5).std()))
    df['rsi_14'] = 100 - (100 / (1 + df['close'].diff().rolling(14).mean() / df['close'].diff().rolling(14).std()))

    return df.dropna()

# ✅ ML Model Initialize
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

# ✅ Training Data Load Karo
def train_model():
    df = fetch_live_data()

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['ema_4', 'ema_8', 'ema_12', 'ema_16', 'ema_20', 'rsi_5', 'rsi_14']
    
    X = df[features].fillna(method='ffill')
    y = df['target']
    
    model.fit(X, y)
    print("✅ Model Training Completed!")

train_model()  # Pehle model train kar lo

# ✅ API Endpoint for Buy/Sell Signal
@app.route('/signal', methods=['GET'])
def get_signal():
    df = fetch_live_data()
    features = ['ema_4', 'ema_8', 'ema_12', 'ema_16', 'ema_20', 'rsi_5', 'rsi_14']
    
    latest_data = df.iloc[-1:][features].fillna(method='ffill')
    prediction = model.predict(latest_data)[0]
    
    signal = "BUY" if prediction == 1 else "SELL"
    
    return jsonify({
        "timestamp": str(df.iloc[-1]['timestamp']),
        "close": df.iloc[-1]['close'],
        "signal": signal
    })

# ✅ Flask Server Run Karo
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)