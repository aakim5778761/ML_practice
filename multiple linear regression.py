import pandas as pd
from sklearn.preprocessing import OneHotEncoder

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data2.csv"
data = pd.read_csv(url)

data["EducationLevel"] = data["EducationLevel"].map({"高中以下": 0, "大學": 1, "碩士以上": 2})


onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[["City"]])
city_encoded = onehot_encoder.transform(data[["City"]]).toarray()


data[["CityA","CityB","CityC"]] = city_encoded

data = data.drop(["City","CityC"],axis=1)


from sklearn.model_selection import train_test_split
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]]
y = data["Salary"]

#固定分割random_state=87
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#直接使用訓練集的轉換結果去轉測試集
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import numpy as np

def compute_cost(x, y, w, b):
  y_pred = (x*w).sum(axis=1) + b
  cost = ((y - y_pred)**2).mean()
  return cost

w = np.array([0, 2, 2, 4])
b = 0
compute_cost(x_train, y_train, w, b)


def compute_gradient(x, y, w, b):
  y_pred = (x*w).sum(axis=1) + b
  #w_gradient開4格且皆為0放w1,w2,w3,w4斜率
  w_gradient = np.zeros(x.shape[1])
  #b的斜率
  b_gradient = (y_pred - y).mean()
  #wi斜率為2*xi(y_pred - y) #把2省略不影響結果
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

    w = w - w_gradient*learning_rate
    b = b - b_gradient*learning_rate
    cost = cost_function(x, y, w, b)

    w_hist.append(w)
    b_hist.append(b)
    c_hist.append(cost)

    if i%p_iter == 0:
      print(f"Iteration {i:5} : Cost {cost: .4e}, w: {w}, b: {b: .2e}, w_gradient: {w_gradient}, b_gradient: {b_gradient: .2e}")

  return w, b, w_hist, b_hist, c_hist


w_init = np.array([1, 2, 2, 4])
b_init = 0
learning_rate = 1.0e-2
run_iter = 10000

w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)

y_pred = (w_final*x_test).sum(axis=1) + b_final
pd.DataFrame({
    "y_pred": y_pred,
    "y_test": y_test
})



# 5.3 碩士以上 城市A
# 7.2 高中以下 城市B
x_real = np.array([[5.3, 2, 1, 0], [7.2, 0, 0, 1]])
x_real = scaler.transform(x_real)
y_real = (w_final*x_real).sum(axis=1) + b_final
y_real



