import pandas as pd
import wget
import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.font_manager import fontManager
import numpy as np

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)

# y = w*x + b
x = data["YearsExperience"]
y = data["Salary"]


fontManager.addfont("ChineseFont.ttf")
mpl.rc('font', family="ChineseFont")


def compute_cost(x, y, w, b):
  y_pred = w*x + b
  cost = (y - y_pred)**2
  cost = cost.sum() / len(x)
  return cost


costs = []
#for w in range(-100, 101):
#  cost = compute_cost(x, y, w, 0)
#  costs.append(cost)

ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))

i = 0
for w in ws: 
  j = 0
  for b in bs: 
    cost = compute_cost(x, y, w, b)
    costs[i,j] = cost
    j = j+1
  i = i+1

plt.figure(figsize=(7, 7))#調整圖片大小
ax = plt.axes(projection="3d")
ax.view_init(0, 10) # 旋轉3d圖 第一個參數是旋轉上下 第二個是左右
ax.xaxis.set_pane_color((0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0))
ax.zaxis.set_pane_color((0, 0, 0))

b_grid, w_grid = np.meshgrid(bs, ws)
# https://wangyeming.github.io/2018/11/12/numpy-meshgrid/

#cmap="Spectral_r" 圖變彩色  alpha改透明度
ax.plot_surface(w_grid, b_grid, costs, cmap="Spectral_r", alpha=0.7)
#plot_wireframe圖加上邊寬
ax.plot_wireframe(w_grid, b_grid, costs, color="black", alpha=0.5)

ax.set_title("w b 對應的 cost")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")

w_index, b_index = np.where(costs == np.min(costs))
#最小cost顯示成紅色 s改變點的大小
ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color="red", s=40)

plt.show()

print(f"當w={ws[w_index]}, b={bs[b_index]} 會有最小cost:{costs[w_index, b_index]}")






