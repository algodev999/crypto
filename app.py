import streamlit as st 
import yfinance as yf
import pandas as pd
import os
import base64

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback


# List of available cryptocurrencies
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOT-USD', 'USDC-USD', 'DOGE-USD', 'AVAX-USD']
crypto_names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'XRP', 'Polkadot', 'USD Coin', 'Dogecoin', 'Avalanche']


#---------------------------------------------------- Functions ------------------------------------

# Function to create a link to download the data file
def create_download_link(csv_path, filename):
    with open(csv_path, "rb") as f:
        content = f.read()
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV ({filename})</a>'
    return href

# Function to download minute data
def download_minute_data(ticker, start, end):
    data_frames = []
    # Split the period into 2-day blocks to try to bypass limitations
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=1, hours=23, minutes=59), end)
        data = yf.download(ticker, start=current_start, end=current_end, interval='1m')
        data_frames.append(data)
        current_start = current_end + timedelta(minutes=1)  # Start one minute after the last end
    # Concatenate all the obtained DataFrames
    return pd.concat(data_frames)


#Clean the csv file due to yfinance API update to Version 0.2.55
def clean_csv_yfinance(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    if len(lines) > 1 and 'Ticker' in lines[1]:
        lines.pop(2)
        lines.pop(1)

    cols = lines[0].strip().split(',')
    cols[0] = 'Datetime'
    lines[0] = ','.join(cols) + '\n'

    with open(csv_path, 'w') as f:
        f.writelines(lines)

# Function to save the data
def download_crypto_data(symbol, name):
    try:
        # Set the total period of 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Create folder for CSV files if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        print(f"Downloading data for {name}")
        data = download_minute_data(symbol, start_date, end_date)
        csv_path = os.path.join('data', f'{name}.csv')
        data.to_csv(csv_path, index=True, encoding='utf-8', sep=',')
        
        st.markdown(create_download_link(csv_path, f"{symbol}_original.csv"), unsafe_allow_html=True)

        clean_csv_yfinance(csv_path)

        

        if not data.empty:
            return True, None
        else:
            return False, "No data was returned."
    except Exception as e:
        return False, str(e)


# Function to load data from a CSV file
def load_crypto_data(name):
    try:
        data = pd.read_csv(f'data/{name}.csv')
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        start_date = data['Datetime'].iloc[0].date()
        end_date = data['Datetime'].iloc[-1].date()
        return data, start_date, end_date
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


# Function to create data sequences
def create_sequences(data, seq_length, forecast_length):
    X = []
    y = []
    for i in range(len(data) - seq_length - forecast_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + forecast_length), 3])  # Predict 'Close' only
    return np.array(X), np.array(y)


# Function to reverse normalization
def invert_normalization(scaler, data, n_features):
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, 3] = data  # Index 3 for 'Close'
    inverted = scaler.inverse_transform(dummy)[:, 3]
    return inverted


# Function to train the GRU model
def train_gru(X_train, y_train, X_val, y_val, seq_length, forecast_length, n_features, epochs):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(GRU(50))
    model.add(Dense(forecast_length))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    streamlit_callback = StreamlitCallback(epochs)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, streamlit_callback])
    return model, history


# Function to generate a download link for a file
def generate_download_link(file_path, file_name):
    with open(file_path, "rb") as f:
        bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download the file {file_name}</a>'


# Callback to update the epoch progress in Streamlit
class StreamlitCallback(Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} - Loss: {logs['loss']:.4f}")


#---------------------------------------------------- Page Objects ------------------------------------


# Streamlit app configuration

# Add a background image to the sidebar
def set_sidebar_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Call the function to set the background image in the sidebar
set_sidebar_background("background.png")

st.markdown("""
    <style>
    .stButton button, .stDownloadButton button {
        width: 100%;
        height: 50px;
    }
    .stSelectbox div[data-baseweb="select"] {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


# Adjust the position of the main title
st.markdown("<h1 style='text-align: center; margin-top: -60px;'>Cryptocurrency Forecast</h1>", unsafe_allow_html=True)


# Layout with a sidebar for options
with st.sidebar:

    st.markdown("<h2 style='margin-top: 60px;'>Options</h2>", unsafe_allow_html=True)


    # Cryptocurrency selection
    crypto = st.selectbox('Select the Cryptocurrency', crypto_names)
    crypto_symbol = cryptos[crypto_names.index(crypto)]

    # Button to download new data
    new_data_button = st.button('Get new data')

    # Button to load existing data
    load_data_button = st.button('Load existing data')

    # Forecast horizon selection
    forecast_options = {'60 minutes': 60, '2 hours': 120, '3 hours': 180, '4 hours': 240}
    forecast_choice = st.selectbox('Select forecast horizon', list(forecast_options.keys()))
    forecast_length = forecast_options[forecast_choice]

    # Button to train and save the model
    train_model_button = st.button('Train model and save')

    # Button to load the model and make predictions
    predict_button = st.button('Load model and make predictions')

    # Button to show instructions and allow download of requirements.txt
    instructions_button = st.button('Instructions')

    # Button to show the app.py code
    show_code_button = st.button('Show App Code')

    # Author text
    st.markdown(
        """
        **Application developed by Miguel Machado as part of the Master's thesis for the Computer Engineering course - Information and Knowledge Systems branch at Instituto Superior de Engenharia do Porto - ISEP. To ensure continuous improvement and address potential issues, users are encouraged to provide feedback via email at crypto.mm.feedback@gmail.com**
        """
    )

    


# Button function to download the data
if new_data_button:
    success, error_message = download_crypto_data(crypto_symbol, crypto)
    if success:
        st.success(f'{crypto} data saved successfully!')
    else:
        st.error(f'Failed to obtain data: {error_message}')


# Button function to load the data
if load_data_button:
    data, start_date, end_date = load_crypto_data(crypto)
    if data is not None:
        st.success(f'Data for {crypto} loaded - period: {start_date} to {end_date}')
        
        # Plot of the Close value of the loaded data
        fig, ax = plt.subplots()
        ax.plot(data['Datetime'], data['Close'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Price in USD per minute for {crypto}')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6)) 
        fig.autofmt_xdate()
        st.pyplot(fig)

# Function for the Button to train the model
if train_model_button:
    data, start_date, end_date = load_crypto_data(crypto)
    if data is not None:
        st.success(f'Data for {crypto} loaded - period: {start_date} to {end_date}')
        
        # Normalization and creation of data sequences
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
        seq_length = forecast_length  # Using one hour of data for prediction
        X, y = create_sequences(data_normalized, seq_length, forecast_length)
        
        # Split data into training and validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(['Open', 'High', 'Low', 'Close', 'Volume'])))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(['Open', 'High', 'Low', 'Close', 'Volume'])))
        
        # Train the GRU model
        epochs = 50  # Number of epochs for training
        with st.spinner('Training the model...'):
            model, history = train_gru(X_train, y_train, X_val, y_val, seq_length, forecast_length, len(['Open', 'High', 'Low', 'Close', 'Volume']), epochs)
        
        # Save the trained model
        model.save(f'models/GRU_{crypto_symbol}_{forecast_length}.h5')
        st.success(f'GRU model for {crypto} with a {forecast_length} minutes forecast trained and saved successfully!')
        
        # Make predictions for the validation set
        predictions_val = model.predict(X_val)
        
        # Reverse the normalization of the predictions and actual values
        predictions_val_original = invert_normalization(scaler, predictions_val[:, -1], len(['Open', 'High', 'Low', 'Close', 'Volume']))
        y_val_original = invert_normalization(scaler, y_val[:, -1], len(['Open', 'High', 'Low', 'Close', 'Volume']))
        
        # Calculate MSE, RMSE, MAE, MAPE
        mse = mean_squared_error(y_val_original, predictions_val_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_original, predictions_val_original)
        mape = mean_absolute_percentage_error(y_val_original, predictions_val_original) * 100
        
        # Normalized RMSE
        normalized_rmse = rmse / (y_val_original.max() - y_val_original.min())
        
        # Present the metrics
        st.subheader('Error Metrics')
        st.write(f'MSE: {mse}')
        st.write(f'RMSE: {rmse}')
        st.write(f'MAE: {mae}')
        st.write(f'MAPE: {mape}%')
        st.write(f'Normalized RMSE: {normalized_rmse}')
        
        # Training and test graph
        st.subheader('Training and Test Graph')
        fig, ax = plt.subplots()
        ax.plot(y_val_original, label='Actual Value')
        ax.plot(predictions_val_original, label='Prediction')
        ax.legend()
        st.pyplot(fig)


# Function for the Prediction Button
if predict_button:
    if os.path.exists(f'models/GRU_{crypto_symbol}_{forecast_length}.h5'):
        # Load the saved model
        model = load_model(f'models/GRU_{crypto_symbol}_{forecast_length}.h5')
        data, start_date, end_date = load_crypto_data(crypto)
        if data is not None:
            st.success(f'Data for {crypto} loaded - period: {start_date} to {end_date}')
            
            # Normalization and creation of data sequences
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_normalized = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
            seq_length = 60  # One hour of data for prediction
            
            # Prediction for the next n minutes
            last_sequence = data_normalized[-seq_length:]
            last_sequence = last_sequence.reshape(1, seq_length, len(['Open', 'High', 'Low', 'Close', 'Volume']))
            predictions_future = model.predict(last_sequence)
            
            # Reverse the normalization of the predictions and actual values
            predictions_future_original = invert_normalization(scaler, predictions_future.flatten(), len(['Open', 'High', 'Low', 'Close', 'Volume']))
            
            # Prediction graph
            st.subheader(f'Prediction for the next {forecast_length} minutes')
            fig, ax = plt.subplots()
            ax.plot(predictions_future_original, label='Future Prediction')
            ax.set_xlabel('Minutes')
            ax.set_ylabel('Price in USD')
            ax.legend()
            st.pyplot(fig)
            
            st.success(f'GRU model for {crypto} loaded and {forecast_length} minutes prediction successfully completed!')
    else:
        st.error(f'GRU model for {crypto} with a {forecast_length} minutes forecast not found. Train and save the model first.')


# Function for the Instructions Button
if instructions_button:
    st.markdown("### Instructions to install the `requirements.txt` file and run Streamlit on a local machine")
    st.markdown("""
        1. **Install the dependencies from the `requirements.txt` file**:
           ```bash
           pip install -r requirements.txt
           ```

        2. **Run the Streamlit application**:
           ```bash
           streamlit run app.py
           ```
    """)
    
    st.markdown("### Contents of the `requirements.txt` file")
    try:
        with open("requirements.txt", "r") as file:
            requirements_content = file.read()
            st.code(requirements_content, language='text')
            download_link = generate_download_link("requirements.txt", "requirements.txt")
            st.markdown(download_link, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("`requirements.txt` file not found.")


# Function for the Show Code Button
if show_code_button:
    st.markdown("### App Code")
    try:
        with open("app.py", "r") as file:
            code_content = file.read()
            st.code(code_content, language='python')
            download_link = generate_download_link("app.py", "app.py")
            st.markdown(download_link, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("`app.py` file not found.")