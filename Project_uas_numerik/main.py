import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Durasi_Olahraga": [30, 60, 45, 20, 50, 10, 70, 40, 65, 35],
    "Langkah_Harian": [3000, 7000, 5000, 2000, 6000, 1000, 8000, 4000, 7500, 3500],
    "Kalori": [1800, 2500, 2200, 1500, 2400, 1300, 2700, 2000, 2600, 1900], 
}

df = pd.DataFrame(data)

X = df[["Durasi_Olahraga", "Langkah_Harian"]]
y = df["Kalori"]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

svr_model = SVR(kernel='rbf')

svr_model.fit(X_train, y_train)

y_pred_scaled = svr_model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("Hasil Prediksi Konsumsi Kalori Harian:\n", y_pred)
print("\nMean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

durasi_olahraga = float(input("Masukkan durasi olahraga (dalam menit): "))
langkah_harian = float(input("Masukkan jumlah langkah harian: "))

input_baru = pd.DataFrame([[durasi_olahraga, langkah_harian]], columns=["Durasi_Olahraga", "Langkah_Harian"])
input_baru_scaled = scaler_X.transform(input_baru)
prediksi_baru_scaled = svr_model.predict(input_baru_scaled)
prediksi_baru = scaler_y.inverse_transform(prediksi_baru_scaled.reshape(-1, 1)).ravel()

print(f"\nPrediksi Kalori untuk input baru (Durasi Olahraga: {durasi_olahraga} menit, Langkah Harian: {langkah_harian}):", prediksi_baru[0])
