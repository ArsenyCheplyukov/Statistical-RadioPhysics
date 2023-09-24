import os
from math import log

import matplotlib.pyplot as plt
import numpy as np

# importing pandas
import pandas as pd
from matplotlib import rcParams
from scipy.optimize import curve_fit

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]


def func(x, a, b, c):
    return np.log2(a * x + c) + b


x = np.array([128, 256, 512, 1024, 2048, 4096])
y = np.array([10, 15, 17, 18, 19, 20])

popt, pcov = curve_fit(func, x, y)
x_curve = np.linspace(np.min(x), np.max(x), 250)
y_curve = func(x_curve, *popt)


plt.plot(x_curve, y_curve, color="blue")  # , label=r"$M_{в}$"
# plt.plot(x, y, color="red", label=r"$M_{ср}$")
plt.scatter(x, y, color="blue", linewidth=3)
# plt.scatter(x, y, color="red", linewidth=3)

# # Add grid, title, axis labels and legend
plt.grid(True)
plt.title(
    "Зависимость разницы максимальных значений\nамплитуды от размера набора кадров",
    fontsize=14,
    linespacing=1,
)
plt.xlabel("Размер кадрирования", fontsize=14, linespacing=1.5)
plt.ylabel("Различия плотности мощности, Дб", fontsize=14, linespacing=1.5)
# plt.legend()

# Show the plot
plt.show()
