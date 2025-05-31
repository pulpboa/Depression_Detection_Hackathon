import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the synthetic data
data = pd.read_csv('C:/Users/HP X360 G2/Desktop/synthetic_typing_data.csv')

# Define features and target
features = ['typing_speed_wpm', 'inter_key_latency_ms', 'backspace_ratio', 'session_duration_s']
X = data[features]
y = data['label']

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

type_speed=float(input("Enter your typing speed in wpm "))
inter_key_latency=float(input("Enter your inter key latency in ms "))
backspace_ratio=float(input("Enter your backspace ratio "))
ses_dur=float(input("Enter your session duration in seconds "))

test_data=[[type_speed, inter_key_latency, backspace_ratio, ses_dur]]

if model.predict(test_data)==[0]:
    print("You might be affected with depression")
else:
    print("You seem healthy!!")