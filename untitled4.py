import pandas as pd

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Classification/Diabetes_Data.csv"
data = pd.read_csv(url)

data["Gender"] = data["Gender"].map({"男生":1,"女生":0})

from sklearn.model_selection import train_test_split
x = data[["Age", "Weight", "BloodSugar", "Gender"]]
y = data["Diabetes"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import numpy as np

def compute_cost(x, y, w, b):
    # 計算預測值
    z = (w*x).sum(axis=1) + b
    y_pred = 1/(1+np.exp(-z))
    #np.set_printoptions(formatter={'float': '{:.2f}'.format})
    #print(y_pred)
    # 計算成本(分數)
    cost = ((y - y_pred)**2).mean()
    return cost

w = np.array([1, 2, 3, 4])
b = 1
compute_cost(x_train, y_train, w, b)

def compute_gradient(x, y, w, b):
    # 計算預測值
    z = (w*x).sum(axis=1) + b
    y_pred = 1/(1+np.exp(-z))
    # 計算斜率
    w_gradient = np.zeros(x.shape[1])
    b_gradient = (y_pred - y).mean()
    for i in range(x.shape[1]):
        w_gradient[i] = (x[:, i]*(y_pred - y)).mean()
    return w_gradient, b_gradient








