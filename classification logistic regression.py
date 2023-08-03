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
    # 計算成本
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


np.set_printoptions(formatter={'float': '{: .2e}'.format})
def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=1000):
    c_hist = []
    w_hist = []
    b_hist = []
    w = w_init
    b = b_init
    for i in range(run_iter):
        w_gradient, b_gradient = gradient_function(x, y, w, b)
        # 更新權重和偏差
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = cost_function(x, y, w, b)
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)
        if i % p_iter == 0:
            print(f"Iteration {i:5} : Cost {cost: .4e}, w: {w}, b: {b: .2e}, w_gradient: {w_gradient}, b_gradient: {b_gradient: .2e}")
    return w, b, w_hist, b_hist, c_hist

w_init = np.array([1, 2, 2, 4])
b_init = 0
learning_rate = 2.0e-2
run_iter = 12000

w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)


z_final = (w_final*x_test).sum(axis=1) + b_final
y_pred = 1/(1+np.exp(-z_final))


answer=[]
for j in range(len(y_pred)):
    if y_pred[j] > 0.21:
        answer.append(1)
    else:
        answer.append(0)

compare=pd.DataFrame({
    "answer": answer,
    "y_test": y_test,
    "y_pred":y_pred
})

compare = compare.reset_index(drop=True)  # 重置索引
miss=0
for j in range(len(compare)):
    if compare["answer"][j] != compare["y_test"][j]:
        miss+=1

accuracy = 1 - (miss/len(y_pred))
print("正確率: " + str(accuracy))













