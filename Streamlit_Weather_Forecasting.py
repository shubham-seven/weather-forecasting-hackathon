import pandas as pd
import numpy as np
import joblib
import requests
import time
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the saved models and scalers
@st.cache_resource
def load_models():
    print("Loading models and scalers...")
    start_time = time.time()
    try:
        temp_model = joblib.load('temp_model_xgb.pkl')
        precip_model = joblib.load('precip_model_xgb.pkl')
        scaler_temp = joblib.load('scaler.pkl')
        scaler_precip = joblib.load('precip_scaler.pkl')
        print(f"Models and scalers loaded in {time.time() - start_time:.2f} seconds")
        return temp_model, precip_model, scaler_temp, scaler_precip
    except Exception as e:
        st.error(f"Error loading models/scalers: {e}")
        raise

temp_model, precip_model, scaler_temp, scaler_precip = load_models()

# Initialize history buffer in session state
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame({
        'temperature': [27.8, 27.5, 27.2],
        'humidity': [68.0, 67.5, 68.0]
    })
    print("History buffer initialized:", st.session_state.history.shape)

# Function to fetch data from Open-Meteo API
@st.cache_data(ttl=3600)
def fetch_weather_data(latitude=19.0728, longitude=72.8826, forecast_days=7):
    print("Fetching weather data from API...")
    start_time = time.time()
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,pressure_msl&forecast_days={forecast_days}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        hourly = data['hourly']
        df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'temperature': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'wind_speed': hourly['wind_speed_10m'],
            'wind_direction': hourly['wind_direction_10m'],
            'pressure': hourly['pressure_msl'],
            'cloud_coverage': hourly['cloud_cover'],
            'precipitation': hourly['precipitation']
        })
        print(f"API data fetched in {time.time() - start_time:.2f} seconds")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        raise
    except Exception as e:
        st.error(f"Error processing API data: {e}")
        raise

# Function to preprocess data with historical lags
def preprocess_data(new_data_df, history=None):
    print("Preprocessing data...")
    start_time = time.time()
    features = ['temperature', 'humidity', 'wind_x', 'wind_y', 'pressure',
                'cloud_coverage', 'has_precipitation', 'hour', 'day_of_week', 'month',
                'temp_lag1', 'temp_lag2', 'temp_lag3', 'humidity_lag1']
    
    try:
        df_new = new_data_df.copy()
        df_new['hour'] = df_new['time'].dt.hour
        df_new['day_of_week'] = df_new['time'].dt.dayofweek
        df_new['month'] = df_new['time'].dt.month
        df_new['has_precipitation'] = (df_new['precipitation'] > 0).astype(int)
        
        df_new['wind_x'] = df_new['wind_speed'] * np.cos(np.radians(df_new['wind_direction']))
        df_new['wind_y'] = df_new['wind_speed'] * np.sin(np.radians(df_new['wind_direction']))
        
        if history is not None and len(history) >= 3:
            temp_history = history['temperature'].tolist()[-3:]
            humid_history = history['humidity'].tolist()[-1:]
            df_new['temp_lag1'] = [temp_history[-1] if i > 0 else df_new['temperature'].iloc[0] for i in range(len(df_new))]
            df_new['temp_lag2'] = [temp_history[-2] if i > 1 else df_new['temperature'].iloc[0] for i in range(len(df_new))]
            df_new['temp_lag3'] = [temp_history[-3] if i > 2 else df_new['temperature'].iloc[0] for i in range(len(df_new))]
            df_new['humidity_lag1'] = [humid_history[-1] if i > 0 else df_new['humidity'].iloc[0] for i in range(len(df_new))]
        else:
            df_new['temp_lag1'] = df_new['temperature']
            df_new['temp_lag2'] = df_new['temperature']
            df_new['temp_lag3'] = df_new['temperature']
            df_new['humidity_lag1'] = df_new['humidity']
        
        print(f"Data preprocessed in {time.time() - start_time:.2f} seconds")
        return df_new[features]
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        raise

# Function to make predictions
def predict_weather(new_data_df, history=None):
    print("Making predictions...")
    start_time = time.time()
    try:
        df_new = preprocess_data(new_data_df, history)
        X_new_temp_scaled = scaler_temp.transform(df_new)
        X_new_precip_scaled = scaler_precip.transform(df_new)
        
        temp_preds = temp_model.predict(X_new_temp_scaled)
        precip_preds = precip_model.predict(X_new_precip_scaled)
        
        result = pd.DataFrame({
            'time': new_data_df['time'],
            'temperature': temp_preds,
            'has_precipitation': precip_preds.astype(bool)
        })
        print(f"Predictions made in {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        raise

# Streamlit app
st.title("🌦️ Mumbai Weather Prediction Dashboard")
st.write("Forecast temperature and precipitation for Mumbai using trained XGBoost models and Open-Meteo API data.")

# Sidebar for settings
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Select forecast days", 1, 7, 7)
update_button = st.sidebar.button("Update Predictions")
save_csv = st.sidebar.checkbox("Save predictions to CSV")

# Main content
if update_button:
    with st.spinner("Fetching data and predicting..."):
        weather_data = fetch_weather_data(forecast_days=forecast_days)
        predictions = predict_weather(weather_data, st.session_state.history)
        
        # Update history
        latest_data = weather_data.iloc[-1].to_dict()
        st.session_state.history = pd.concat([
            st.session_state.history,
            pd.DataFrame([{'temperature': latest_data['temperature'], 'humidity': latest_data['humidity']}])
        ], ignore_index=True)
        if len(st.session_state.history) > 3:
            st.session_state.history = st.session_state.history.iloc[1:]
        
        # Save to CSV if checked
        if save_csv:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            predictions.to_csv(f'weather_forecast_{timestamp}.csv', index=False)
            st.success(f"Predictions saved to weather_forecast_{timestamp}.csv")
        
        # Display metrics
        st.subheader("Current Conditions")
        col1, col2 = st.columns(2)
        col1.metric("Temperature", f"{weather_data['temperature'].iloc[-1]:.1f} °C")
        col2.metric("Precipitation", f"{'Yes' if weather_data['precipitation'].iloc[-1] > 0 else 'No'}")
        
        # Display chart
        st.subheader(f"{forecast_days}-Day Temperature Forecast")
        st.line_chart(predictions.set_index('time')['temperature'])
        
        # Display table
        st.subheader("Prediction Details")
        st.dataframe(predictions.style.format({
            'time': lambda x: x.strftime('%Y-%m-%d %H:%M'),
            'temperature': '{:.2f}',
            'has_precipitation': lambda x: 'Yes' if x else 'No'
        }))
else:
    st.info("Click 'Update Predictions' to load the latest forecast.")