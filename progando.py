import numpy as np
import pandas as pd
import logging
import os
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression

# Disable oneDNN in TensorFlow to avoid numerical errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class SmartTraderPredictor:
    def __init__(self, historical_data_path):
        print("Iniciando SmartTraderPredictor...")
        self.historical_data_path = historical_data_path
        self.driver = None
        self.last_number = None
        self.historical_digits = []
        self.lstm_model = None
        self.rf_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.service = Service(ChromeDriverManager().install())
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='smarttrader_predictor.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def setup_driver(self):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

            self.driver = webdriver.Chrome(service=self.service, options=options)
            self.driver.get('https://smarttrader.deriv.com/es/trading?currency=USD&market=synthetics&underlying=R_100&formname=matchdiff&duration_amount=1&duration_units=t&amount=315&amount_type=stake&expiry_type=duration&multiplier=1&prediction=8')
            logging.info("WebDriver initialized successfully")
            print("WebDriver configurado correctamente")
        except Exception as e:
            logging.error(f"Error initializing WebDriver: {e}")
            print(f"Error inicializando WebDriver: {e}")
            raise

    def read_historical_data(self):
        data = []
        count = 0

        def process_chunk(chunk):
            local_data = []
            local_count = 0
            for line in chunk:
                if "Número capturado" in line:
                    try:
                        number_str = line.split(" - ")[0].replace("Número capturado: ", "").strip().replace(".", "")
                        if re.match(r'^\d+$', number_str):
                            local_data.append(int(number_str))
                            local_count += 1
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error parsing line: {line.strip()}. Error: {e}")
                        print(f"Error parseando línea: {line.strip()}. Error: {e}")
            return local_data, local_count

        try:
            print("Leyendo datos históricos...")
            
            chunk_size = 100000
            with open(self.historical_data_path, 'r', buffering=10**7) as file:
                while chunk := list(file.readlines(chunk_size)):
                    with ThreadPoolExecutor() as executor:
                        results = list(executor.map(process_chunk, [chunk]))
                    for local_data, local_count in results:
                        data.extend(local_data)
                        count += local_count

            logging.info(f"Total historical numbers read: {len(data)}")
            logging.info(f"Total lines processed: {count}")
            print(f"Total de números históricos leídos: {len(data)}")
        except Exception as e:
            logging.error(f"Error reading historical data: {e}")
            print(f"Error leyendo datos históricos: {e}")
        return data

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(4, activation='relu', input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(2, activation='relu', return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(4, activation='relu'),
            Dense(1)  # Cambiado a 1 salida para regresión
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Cambiado loss y metrics para regresión
        return model

    def optimize_random_forest(self, X, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def ensemble_models(self, X, y):
        try:
            print("Realizando predicciones con LSTM...")
            lstm_pred = self.lstm_model.predict(X)
            print("Predicción con LSTM completada.")
            print("Realizando predicciones con Random Forest...")
            rf_pred = self.rf_model.predict(X).reshape(-1, 1)
            print("Predicción con Random Forest completada.")
            print("Realizando predicciones con XGBoost...")
            xgb_pred = self.xgb_model.predict(X).reshape(-1, 1)
            print("Predicción con XGBoost completada.")
            print("Ensamblando resultados...")
            combined_pred = np.hstack((lstm_pred, rf_pred, xgb_pred))
            meta_model = LinearRegression()
            meta_model.fit(combined_pred, y)
            print("Ensamblaje completado.")
            return meta_model
        except Exception as e:
            logging.error(f"Error during model ensembling: {e}")
        print(f"Error durante el ensamblaje de modelos: {e}")

    def validate_model(self, model, X, y):
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -scores  # Convertir a MSE positivo
        rmse_scores = np.sqrt(mse_scores)
        logging.info(f"Cross-validation RMSE: {np.mean(rmse_scores)}")
        print(f"RMSE de validación cruzada: {np.mean(rmse_scores)}")

    def prepare_sequence_data(self, data, sequence_length):
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(np.array(data).reshape(-1, 1))
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y).ravel()  # Flatten y

    def train_models(self):
        print("Entrenando modelos...")
        self.historical_digits = self.read_historical_data()

        if len(self.historical_digits) < 100:
            logging.warning("No hay suficientes datos históricos para entrenar los modelos")
            print("No hay suficientes datos históricos para entrenar los modelos")
            return

        sequence_length = 50
        X, y = self.prepare_sequence_data(self.historical_digits, sequence_length)

        # LSTM model
        input_shape = (X.shape[1], 1)
        self.lstm_model = self.build_lstm_model(input_shape)
        self.lstm_model.fit(X, y, epochs=1, batch_size=128, validation_split=0.1, callbacks=[EarlyStopping(patience=3)])

        # Random Forest model
        X_2d = X.reshape(X.shape[0], -1)
        self.rf_model = self.optimize_random_forest(X_2d, y)

        # XGBoost model
        self.xgb_model = XGBRegressor()
        self.xgb_model.fit(X_2d, y)

        # Ensemble model
        self.ensemble_model = self.ensemble_models(X, y)
        self.validate_model(self.ensemble_model, X_2d, y)

        logging.info("Entrenamiento de modelos completado.")
        print("Entrenamiento completado.")

# Crear una instancia del predictor y entrenar los modelos
smart_trader_predictor = SmartTraderPredictor('D:/Escritorio/cinta holografica/reporte_numeros.txt')
smart_trader_predictor.setup_driver()
smart_trader_predictor.train_models()
