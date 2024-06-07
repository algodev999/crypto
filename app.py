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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Lista de criptomoedas disponíveis
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOT-USD', 'USDC-USD', 'DOGE-USD', 'AVAX-USD']
crypto_names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'XRP', 'Polkadot', 'USD Coin', 'Dogecoin', 'Avalanche']






#---------------------------------------------------- Funções ------------------------------------


# Função para fazer download dos dados ao minuto
def download_minute_data(ticker, start, end):
    data_frames = []
    # Dividir o período em blocos de 2 dias para tentar contornar limitações
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=1, hours=23, minutes=59), end)
        data = yf.download(ticker, start=current_start, end=current_end, interval='1m')
        data_frames.append(data)
        current_start = current_end + timedelta(minutes=1)  # Começar um minuto após o último fim
    # Concatenar todos os DataFrames obtidos
    return pd.concat(data_frames)


# Função para gravação dos dados
def download_crypto_data(symbol, name):
    try:
        # Definir o período total de 30 dias
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Criar pasta para os arquivos CSV, se não existir
        if not os.path.exists('data'):
            os.makedirs('data')
        
        print(f"Downloading data for {name}")
        data = download_minute_data(symbol, start_date, end_date)
        csv_path = os.path.join('data', f'{name}.csv')
        data.to_csv(csv_path)

        if not data.empty:
            return True, None
        else:
            return False, "Nenhum dado foi retornado."
    except Exception as e:
        return False, str(e)


# Função para carregar os dados de um arquivo CSV
def load_crypto_data(name):
    try:
        data = pd.read_csv(f'data/{name}.csv')
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        start_date = data['Datetime'].iloc[0].date()
        end_date = data['Datetime'].iloc[-1].date()
        return data, start_date, end_date
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None


# Função para criar as sequências de dados
def create_sequences(data, seq_length, forecast_length):
    X = []
    y = []
    for i in range(len(data) - seq_length - forecast_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + forecast_length), 3])  # Previsão apenas de 'Close'
    return np.array(X), np.array(y)


# Função para inverter a normalização
def invert_normalization(scaler, data, n_features):
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, 3] = data  # Índice 3 para 'Close'
    inverted = scaler.inverse_transform(dummy)[:, 3]
    return inverted


# Função para treinar o modelo GRU
def train_gru(X_train, y_train, X_val, y_val, seq_length, forecast_length, n_features):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(GRU(50))
    model.add(Dense(forecast_length))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return model, history


# Função para gerar um link de download para um arquivo
def generate_download_link(file_path, file_name):
    with open(file_path, "rb") as f:
        bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Fazer o download do ficheiro {file_name}</a>'











#---------------------------------------------------- Objectos da Página ------------------------------------

# Configuração da aplicação Streamlit

# Adicionar uma imagem de fundo à sidebar
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


# Chamar a função para definir a imagem de fundo na sidebar
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


# Ajuste da posição do título principal
st.markdown("<h1 style='text-align: center; margin-top: -60px;'>Previsão de Criptomoedas</h1>", unsafe_allow_html=True)


# Layout com barra lateral para as opções
with st.sidebar:

    st.markdown("<h2 style='margin-top: 60px;'>Opções</h2>", unsafe_allow_html=True)


    # Seleção da criptomoeda
    crypto = st.selectbox('Selecione a Criptomoeda', crypto_names)
    crypto_symbol = cryptos[crypto_names.index(crypto)]

    # Botão para download dos dados
    new_data_button = st.button('Obter novos dados para treino')

    # Botão para carregar os dados existentes
    load_data_button = st.button('Carregar dados existentes')

    # Seleção do horizonte de previsão
    forecast_options = {'60 minutos': 60, '6 horas': 360, '12 horas': 720, '24 horas': 1440}
    forecast_choice = st.selectbox('Selecione o horizonte de previsão', list(forecast_options.keys()))
    forecast_length = forecast_options[forecast_choice]

    # Botão para treinar o modelo
    train_model_button = st.button('Treinar modelo e gravar')

    # Botão para carregar o modelo e fazer previsões
    predict_button = st.button('Carregar modelo e fazer previsões')

    # Botão para mostrar instruções e permitir download do requirements.txt
    instructions_button = st.button('Instruções')

    # Botão para mostrar o código do arquivo app.py
    show_code_button = st.button('Mostrar código da App')

    # Texto do autor
    st.markdown(
        """
        **Aplicação desenvolvida por Mguel Machado no âmbito da tese de Mestrado do curso de Engenharia Informática - Ramo Sistemas de Informação e Conhecimento do Instituto Superior de Engenharia do Porto - ISEP**
        """
    )


# Função do Botão para fazer download dos dados
if new_data_button:
    success, error_message = download_crypto_data(crypto_symbol, crypto)
    if success:
        st.success(f'Dados da {crypto} gravados com sucesso!')
    else:
        st.error(f'Falha ao obter os dados: {error_message}')


# Função do Botão para carregar os dados
if load_data_button:
    data, start_date, end_date = load_crypto_data(crypto)
    if data is not None:
        st.success(f'Dados para a moeda {crypto} carregados - período: {start_date} a {end_date}')
        
        # Gráfico do valor de Close dos dados carregados
        fig, ax = plt.subplots()
        ax.plot(data['Datetime'], data['Close'])
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.set_title(f'Gráfico do valor em USD ao minuto para {crypto}')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6)) 
        fig.autofmt_xdate()
        st.pyplot(fig)


# Função do Botão para treinar o modelo
if train_model_button:
    data, start_date, end_date = load_crypto_data(crypto)
    if data is not None:
        st.success(f'Dados para a moeda {crypto} carregados - período: {start_date} a {end_date}')
        
        # Normalização e criação de sequências de dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
        seq_length = 60  # Uma hora de dados para previsão
        X, y = create_sequences(data_normalized, seq_length, forecast_length)
        
        # Dividir dados em treino e validação
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(['Open', 'High', 'Low', 'Close', 'Volume'])))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(['Open', 'High', 'Low', 'Close', 'Volume'])))
        
        # Treinar o modelo GRU
        with st.spinner('A treinar o modelo...'):
            model, history = train_gru(X_train, y_train, X_val, y_val, seq_length, forecast_length, len(['Open', 'High', 'Low', 'Close', 'Volume']))
        
        # Gravar o modelo treinado
        model.save(f'models/GRU_{crypto_symbol}_{forecast_length}.h5')
        st.success(f'Modelo GRU para {crypto} com previsão de {forecast_length} minutos treinado e gravado com sucesso!')
        
        # Fazer previsões para o conjunto de validação
        predictions_val = model.predict(X_val)
        
        # Inverter a normalização das previsões e dos valores reais
        predictions_val_original = invert_normalization(scaler, predictions_val[:, -1], len(['Open', 'High', 'Low', 'Close', 'Volume']))
        y_val_original = invert_normalization(scaler, y_val[:, -1], len(['Open', 'High', 'Low', 'Close', 'Volume']))
        
        # Calcular MSE, RMSE, MAE, MAPE
        mse = mean_squared_error(y_val_original, predictions_val_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_original, predictions_val_original)
        mape = np.mean(np.abs((y_val_original - predictions_val_original) / y_val_original)) * 100
        
        # Normalized RMSE
        normalized_rmse = rmse / (y_val_original.max() - y_val_original.min())
        
        # Apresentar as métricas
        st.subheader('Métricas de Erro')
        st.write(f'MSE: {mse}')
        st.write(f'RMSE: {rmse}')
        st.write(f'MAE: {mae}')
        st.write(f'MAPE: {mape}%')
        st.write(f'Normalized RMSE: {normalized_rmse}')
        
        # Gráfico de treino e teste
        st.subheader('Gráfico de Treino e Teste')
        fig, ax = plt.subplots()
        ax.plot(y_val_original, label='Valor Real')
        ax.plot(predictions_val_original, label='Previsão')
        ax.set_xlabel('Minutos')
        ax.set_ylabel('Preço em USD')
        ax.legend()
        st.pyplot(fig)


# Função do Botão de previsão
if predict_button:
    if os.path.exists(f'models/GRU_{crypto_symbol}_{forecast_length}.h5'):
        # Carregar o modelo gravado
        model = load_model(f'models/GRU_{crypto_symbol}_{forecast_length}.h5')
        data, start_date, end_date = load_crypto_data(crypto, )
        if data is not None:
            st.success(f'Dados para a moeda {crypto} carregados - período: {start_date} a {end_date}')
            
            # Normalização e criação de sequências de dados
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_normalized = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
            seq_length = 60  # Uma hora de dados para previsão
            
            # Previsão para os próximos n minutos
            last_sequence = data_normalized[-seq_length:]
            last_sequence = last_sequence.reshape(1, seq_length, len(['Open', 'High', 'Low', 'Close', 'Volume']))
            predictions_future = model.predict(last_sequence)
            
            # Inverter a normalização das previsões e dos valores reais
            predictions_future_original = invert_normalization(scaler, predictions_future.flatten(), len(['Open', 'High', 'Low', 'Close', 'Volume']))
            
            # Gráfico de previsão
            st.subheader(f'Previsão para os próximos {forecast_length} minutos')
            fig, ax = plt.subplots()
            ax.plot(predictions_future_original, label='Previsão Futura')
            ax.set_xlabel('Minutos')
            ax.set_ylabel('Preço em USD')
            ax.legend()
            st.pyplot(fig)
            
            st.success(f'Modelo GRU para {crypto} carregado e previsão de {forecast_length} minutos realizada com sucesso!')
    else:
        st.error(f'Modelo GRU para {crypto} com previsão de {forecast_length} minutos não encontrado. Treine e grave o modelo primeiro.')


# Função do Botão de Instruções
if instructions_button:
    st.markdown("### Instruções para instalar o arquivo `requirements.txt` e executar o Streamlit numa máquina local")
    st.markdown("""
        1. **Instale as dependências a partir do arquivo `requirements.txt`**:
           ```bash
           pip install -r requirements.txt
           ```

        2. **Execute a aplicação Streamlit**:
           ```bash
           streamlit run app.py
           ```
    """)
    
    st.markdown("### Conteúdo do arquivo `requirements.txt`")
    try:
        with open("requirements.txt", "r") as file:
            requirements_content = file.read()
            st.code(requirements_content, language='text')
            download_link = generate_download_link("requirements.txt", "requirements.txt")
            st.markdown(download_link, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Arquivo `requirements.txt` não encontrado.")


# Função do Botão de apresentação do código da App
if show_code_button:
    st.markdown("### Código da App")
    try:
        with open("app.py", "r") as file:
            code_content = file.read()
            st.code(code_content, language='python')
            download_link = generate_download_link("app.py", "app.py")
            st.markdown(download_link, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Arquivo `app.py` não encontrado.")