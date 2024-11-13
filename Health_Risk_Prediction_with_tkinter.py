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
import tkinter as tk
from tkinter import messagebox
import webbrowser  # For opening the map in a web browser

# API Keys
visual_crossing_api_key = 'YOUR_VISUAL_CROSSING_API_KEY'
openweather_api_key = 'YOUR_OPENWEATHERMAP_API_KEY'

# Functions to fetch data
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

# GUI Class
class HealthRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Health Risk Prediction")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # Rounded corner frame
        self.frame = tk.Frame(self.root, bg="#ffffff", bd=5, relief="groove")
        self.frame.place(relx=0.5, rely=0.5, anchor='center', width=400, height=300)

        self.label_city = tk.Label(self.frame, text="Enter City:", bg="#ffffff", font=("Helvetica", 16))
        self.label_city.pack(pady=10)

        self.entry_city = tk.Entry(self.frame, font=("Helvetica", 14))
        self.entry_city.pack(pady=10)

        self.button_predict = tk.Button(self.frame, text="Predict Health Risk", command=self.predict_health_risk, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.button_predict.pack(pady=20)

        self.text_output = tk.Text(self.frame, height=10, width=50, font=("Helvetica", 12))
        self.text_output.pack(pady=10)

    def clear_output(self):
        self.text_output.delete(1.0, tk.END)

    def predict_health_risk(self):
        self.clear_output()
        city = self.entry_city.get()

        if not city:
            messagebox.showerror("Input Error", "Please enter a city name.")
            return

        lat, lon = fetch_geo_location(city)
        if lat is None or lon is None:
            messagebox.showerror("Location Error", "Could not fetch geolocation data. Please check the city name.")
            return

        # Open map in a web browser
        webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")

        # Fetch environmental data
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        environmental_data = []

        for date in dates:
            env_data = fetch_environmental_data(lat, lon, date)
            if env_data:
                environmental_data.append({
                    'city': city,
                    'date': date,
                    'pm2_5': float(env_data.get('pm2_5', 0) or 0),  # Convert to float
                    'pm10': float(env_data.get('pm10', 0) or 0),    # Convert to float
                    'aqi_us': float(env_data.get('aqi_us', 0) or 0), # Convert to float
                    'aqi_eu': float(env_data.get('aqi_eu', 0) or 0), # Convert to float
                    'temperature': float(env_data.get('temperature', 0) or 0), # Convert to float
                    'humidity': float(env_data.get('humidity', 0) or 0),       # Convert to float
                    'uv_index': float(env_data.get('uv_index', 0) or 0)        # Convert to float
                })

        env_df = pd.DataFrame(environmental_data)
        data = env_df.dropna()

        if len(data) < 2:
            messagebox.showerror("Data Error", "Not enough data to train the model.")
            return

        # Display environmental data in a table-like format
        self.display_environmental_data(data)

    def display_environmental_data(self, data):
        # Create a new window for displaying environmental data
        env_window = tk.Toplevel(self.root)
        env_window.title("Environmental Data")
        env_window.geometry("400x300")
        env_window.configure(bg="#f0f0f0")

        # Create a table-like structure
        for i, column in enumerate(data.columns):
            tk.Label(env_window, text=column, font=("Helvetica", 12, "bold"), bg="#f0f0f0").grid(row=0, column=i)

        for i, row in data.iterrows():
            for j, value in enumerate(row):
                color = self.get_color_based_on_risk(value, j)
                label = tk.Label(env_window, text=value, bg=color, font=("Helvetica", 12))
                label.grid(row=i + 1, column=j)

    def get_color_based_on_risk(self, value, index):
        # Define risk levels based on index (you can adjust the logic as needed)
        if index in [2, 3]:  # pm2_5 and pm10
            if value > 100:
                return "red"  # High risk
            elif value > 50:
                return "orange"  # Medium risk
            else:
                return "yellow"  # Low risk
        else:
            return "green"  # No risk for other parameters

# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthRiskApp(root)
    root.mainloop()
