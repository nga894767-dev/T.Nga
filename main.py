import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv("creditcard.csv")

X = data[["Amount", "Time"]]

model = IsolationForest(contamination=0.01)

model.fit(X)

data["KetQua"] = model.predict(X)

bat_thuong = data[data["KetQua"] == -1]

print("Các giao dịch bất thường:")
print(bat_thuong[["Time", "Amount", "KetQua"]].head())

print("Số giao dịch bất thường:")
print(len(bat_thuong))