import pandas as pd
import wget
import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.font_manager import fontManager
from ipywidgets import interact

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)

# y = w*x + b
x = data["YearsExperience"]
y = data["Salary"]

wget.download("https://github.com/GrandmaCan/ML/raw/main/Resgression/ChineseFont.ttf")

fontManager.addfont("ChineseFont.ttf")
mpl.rc('font', family="ChineseFont")

def plot_pred(w, b):
  y_pred = x*w + b
  plt.plot(x, y_pred, color="blue", label="預測線")
  plt.scatter(x, y, marker="x", color="red", label="真實數據")
  plt.title("年資-薪水")
  plt.xlabel("年資")
  plt.ylabel("月薪(千)")
  plt.xlim([0, 12])
  plt.ylim([-60, 140])
  plt.legend()
  plt.show()

plot_pred(10, 20)

#interact(plot_pred, w=(-100, 100, 1), b=(-100, 100, 1))



