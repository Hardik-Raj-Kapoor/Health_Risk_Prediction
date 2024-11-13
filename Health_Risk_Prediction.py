import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta


visual_crossing_api_key = 'YOUR_VISUAL_CROSSING_API_KEY'
openweather_api_key = 'ec995a56bc555cbcdfd392f33705e66e'


def fetch_geo_location(city):
    api_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}'
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if data and 'coord' in data:
            return data['coord']['lat'], data['coord']['lon']
        return None, None
    except Exception as e:
        print(f"Error fetching geolocation data: {e}")
        return None, None


def fetch_environmental_data(lat, lon, date):
    api_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date}?unitGroup=metric&key={visual_crossing_api_key}&contentType=json&elements=datetime,pm1,pm2p5,pm10,o3,no2,so2,co,aqius,aqieur,temp,humidity,uvindex"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if 'days' in data and data['days']:
            day_data = data['days'][0]
            return {
                'pm2_5': day_data.get('pm2p5', None),
                'pm10': day_data.get('pm10', None),
                'aqi_us': day_data.get('aqius', None),
                'aqi_eu': day_data.get('aqieur', None),
                'temperature': day_data.get('temp', None),
                'humidity': day_data.get('humidity', None),
                'uv_index': day_data.get('uvindex', None)
            }
        return {}
    except Exception as e:
        print(f"Error fetching environmental data: {e}")
        return {}


city = input("Enter a city for health risk prediction: ")
lat, lon = fetch_geo_location(city)
if lat is not None and lon is not None:
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
    environmental_data = []

    for date in dates:
        env_data = fetch_environmental_data(lat, lon, date)
        if env_data:
            environmental_data.append({
                'city': city,
                'date': date,
                'pm2_5': env_data.get('pm2_5'),
                'pm10': env_data.get('pm10'),
                'aqi_us': env_data.get('aqi_us'),
                'aqi_eu': env_data.get('aqi_eu'),
                'temperature': env_data.get('temperature'),
                'humidity': env_data.get('humidity'),
                'uv_index': env_data.get('uv_index')
            })

    env_df = pd.DataFrame(environmental_data)


    data = env_df
    data.dropna(inplace=True)

    if len(data) < 2:
        print("Not enough data to train the model.")
    else:
        print("Collected Environmental Data:")
        print(data)

        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()

        selected_features = ['pm2_5', 'pm10', 'temperature', 'humidity', 'uv_index']
        X = data[selected_features]

        for feature in selected_features:
            for lag in range(1, 3):
                X[f'{feature}_lag{lag}'] = X[feature].shift(lag)

        X.dropna(inplace=True)


        y = X['pm2_5'] * 0.4 + X['pm10'] * 0.3 + X['temperature'] * 0.1 + X['humidity'] * 0.1 + X['uv_index'] * 0.1


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stopping])

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
#        print(f'Deep Learning Mean Squared Error: {mse}')
        current_env_data = fetch_environmental_data(lat, lon, datetime.now().strftime('%Y-%m-%d'))

        print("\n--- Current Weather ---")
        temperature = current_env_data.get('temperature', None)
        humidity = current_env_data.get('humidity', None)
        uv_index = current_env_data.get('uv_index', None)

        print(f"Temperature: {temperature:.2f} °C" if temperature is not None else "Temperature: N/A")
        print(f"Humidity: {humidity}% (N/A)" if humidity is not None else "Humidity: N/A")
        print(f"UV Index: {uv_index:.2f}" if uv_index is not None else "UV Index: N/A")

        input_data = {
            'pm2_5': current_env_data.get('pm2_5'),
            'pm10': current_env_data.get('pm10'),
            'temperature': current_env_data.get('temperature'),
            'humidity': current_env_data.get('humidity'),
            'uv_index': current_env_data.get('uv_index')
        }

        for feature in selected_features:
            for lag in range(1, 3):
                input_data[f'{feature}_lag{lag}'] = input_data[feature]

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        predicted_risk_index = model.predict(input_scaled)[0][0]
        predicted_risk_index = max(0, predicted_risk_index)

        def classify_disease(risk_index):
              if risk_index < 50:
                  classification = "Low risk"
                  advice = [
                      "Maintain a balanced diet rich in fruits and vegetables.",
                      "Engage in regular physical activity, aiming for at least 150 minutes of moderate exercise each week.",
                      "Stay hydrated by drinking plenty of water throughout the day.",
                      "Get sufficient sleep each night (7-9 hours) to help your body recover.",
                      "Consider regular health check-ups to monitor overall well-being.",
                      "Limit processed foods and added sugars to maintain good health.",
                      "Practice mindfulness or meditation to reduce stress.",
                      "Stay informed about air quality and continue healthy habits."
                  ]
                  return classification, advice
              elif 50 <= risk_index < 100:
                  classification = "Moderate risk"
                  advice = [
                      "Exercise regularly but avoid strenuous outdoor activities on high pollution days.",
                      "Consider wearing masks during times of high air pollution or if you have respiratory issues.",
                      "Keep windows closed and use air purifiers indoors to improve air quality.",
                      "Monitor local air quality reports and adjust outdoor activities accordingly.",
                      "Incorporate foods rich in antioxidants, like berries and nuts, into your diet to combat oxidative stress.",
                      "Engage in relaxation techniques, such as yoga or deep breathing, to manage stress levels.",
                      "Stay hydrated and consider herbal teas known for their respiratory benefits, such as peppermint or ginger.",
                      "Plan outdoor activities for times when air quality is better (early morning or late evening)."
                  ]
                  return classification, advice
              elif 100 <= risk_index < 150:
                  classification = "High risk"
                  advice = [
                      "Limit outdoor activities and consider indoor exercises, such as yoga or home workouts.",
                      "Consult a healthcare provider about potential respiratory issues and preventive measures.",
                      "Use a high-efficiency particulate air (HEPA) filter in your home to reduce indoor pollutants.",
                      "Pay attention to your body's response to air quality changes and seek medical advice if symptoms worsen.",
                      "Incorporate anti-inflammatory foods into your diet, such as fatty fish and leafy greens.",
                      "Keep emergency medications for respiratory issues handy, such as inhalers.",
                      "Consider joining a support group for individuals with respiratory conditions for shared advice and experiences.",
                      "Educate yourself on environmental health issues and advocate for cleaner air policies in your community."
                  ]
                  return classification, advice
              else:
                  classification = "Critical risk"
                  advice = [
                      "Seek immediate medical attention if experiencing severe respiratory distress.",
                      "Limit all outdoor activities, especially exercise, until air quality improves significantly.",
                      "Consult with a healthcare professional about the best management plan for any existing respiratory conditions.",
                      "Use an air quality app to stay updated on real-time pollution levels and adjust your activities accordingly.",
                      "Ensure that your living environment is as clean as possible by frequently cleaning surfaces and using air purifiers.",
                      "Follow a personalized health plan as recommended by your healthcare provider, which may include medications or lifestyle changes.",
                      "Consider telehealth consultations if visiting a healthcare facility is difficult due to air quality issues.",
                      "Engage with community organizations focused on health and environmental advocacy to seek support and promote awareness."
                  ]
                  return classification, advice

        def calculate_disease_probabilities(risk_index):
            if risk_index < 50:
                base_prob = 0
            elif 50 <= risk_index < 100:
                base_prob = 25
            elif 100 <= risk_index < 150:
                base_prob = 50
            else:
                base_prob = 75
            probabilities = {
                'COPD': max(0, min(base_prob + 5, 100)),  # Adding a small variation to each disease
                'Emphysema': max(0, min(base_prob + 10, 100)),
                'Heart Disease': max(0, min(base_prob - 5, 100)),
                'Pulmonary Hypertension': max(0, min(base_prob, 100)),
                'Respiratory Failure': max(0, min(base_prob + 10, 100)),
                'Severe Asthma Attacks': max(0, min(base_prob + 5, 100)),
                'Lung Cancer': max(0, min(base_prob - 10, 100))  # Some diseases might have slightly lower probabilities
            }

            return probabilities


        disease_classification, recommendation = classify_disease(predicted_risk_index)
        disease_probabilities = calculate_disease_probabilities(predicted_risk_index)

        print(f"\nPredicted Health Risk Index: {predicted_risk_index:.2f}")
        print(f"Disease Classification: {disease_classification}")
        print(f"Recommendation: {recommendation}")

        print("\n--- Disease Probabilities ---")
        for disease, probability in disease_probabilities.items():
            print(f"{disease}: {probability:.2f}%")


        future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 5)]
        future_environmental_data = []

        for date in future_dates:
            future_env_data = fetch_environmental_data(lat, lon, date)
            if future_env_data:
                future_environmental_data.append({
                    'date': date,
                    'pm2_5': future_env_data.get('pm2_5'),
                    'pm10': future_env_data.get('pm10'),
                    'temperature': future_env_data.get('temperature'),
                    'humidity': future_env_data.get('humidity'),
                    'uv_index': future_env_data.get('uv_index')
                })

        future_df = pd.DataFrame(future_environmental_data)

        combined_df = pd.concat([data[['date', 'pm2_5', 'pm10', 'temperature', 'humidity', 'uv_index']], future_df], ignore_index=True)
        combined_df.reset_index(drop=True, inplace=True)

        X_future = future_df[selected_features]

        last_known_values = data[selected_features].iloc[-2:].reset_index(drop=True)
        for feature in selected_features:
            X_future[f'{feature}_lag1'] = last_known_values[feature].iloc[-1]
            X_future[f'{feature}_lag2'] = last_known_values[feature].iloc[-2]

        X_future_scaled = scaler.transform(X_future)

        future_predictions = model.predict(X_future_scaled)
        future_predictions = [max(0, pred[0]) for pred in future_predictions]

        print("\n--- Predicted Health Risk Index for the Next 4 Days ---")
        for date, risk_index in zip(future_dates, future_predictions):
            print(f"{date}: {risk_index:.2f}")

        historical_dates = data['date'][-len(y):].tolist()
        historical_risk_index = y.tolist()

        plt.figure(figsize=(12, 6))
        plt.plot(historical_dates, historical_risk_index, label='Historical Health Risk Index')
        plt.plot(future_dates, future_predictions, label='Predicted Health Risk Index', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Health Risk Index')
        plt.title(f'Health Risk Index Trends for {city}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))

        colors = sns.color_palette("husl", len(selected_features))

        for i, feature in enumerate(selected_features):
            plt.plot(combined_df['date'], combined_df[feature], label=feature, color=colors[i], linewidth=2, marker='o', markersize=5)

        plt.grid(visible=True, linestyle='--', alpha=0.7)

        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)

        plt.title(f'Environmental Factors Over Time for {city}', fontsize=16, fontweight='bold')

        plt.legend(title='Features', title_fontsize='13', fontsize='11', loc='upper left')

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()
