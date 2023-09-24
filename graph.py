import os

import matplotlib.pyplot as plt
import numpy as np

# importing pandas
import pandas as pd
from matplotlib import rcParams
from scipy.optimize import curve_fit

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]

# GETTING FILE NAMES
first_file_list = []
second_file_list = []

for dirname, _, filenames in os.walk("./"):
    for filename in filenames:
        if filename.startswith("custom_first"):
            first_file_list.append(os.path.join(dirname, filename))
        if filename.startswith("custom_second"):
            second_file_list.append(os.path.join(dirname, filename))

# print(first_file_list)


def func(x, a, b, c):
    return np.log(a * x + c) + b


# функция для получения начала имени файла
def get_prefix(filenames):
    numbers = []
    for file in filenames:
        # разделение имени файла и числа
        name, number = file.split("_")[:-1], file.split("_")[-1]
        # извлечение числа из расширения файла
        number = int(number.split(".")[0])
        # добавление числа в список
        numbers.append(number)
    return numbers


def get_deltas(filenames):
    deltas = []
    for file in filenames:
        # read text file into pandas DataFrame
        df = pd.read_csv(file, sep="\t")
        print(f"for graph {file}:")
        arr = df.iloc[:, 1].nlargest(3)
        print(arr.iloc[0], "\t", arr.iloc[2], "\t", abs(arr.iloc[0] - arr.iloc[2]))
        deltas.append(abs(arr.iloc[2] - arr.iloc[0]))
    return deltas


# list_of_sample_sizes = ["128", "256", "512", "1024", "2048", "4096"]

indexes_of_samples_first = get_prefix(first_file_list)
indexes_of_samples_second = get_prefix(second_file_list)

values_of_samples_first = get_deltas(first_file_list)
values_of_samples_second = get_deltas(second_file_list)
# FIND THE DIFFERENCE BETWEEN TWO NEAREST MAXIMAS

popt, pcov = curve_fit(func, indexes_of_samples_second, values_of_samples_second)
x_curve = np.linspace(np.min(indexes_of_samples_second), np.max(indexes_of_samples_second), 250)
y_curve = func(x_curve, *popt)
plt.plot(x_curve, y_curve, color="blue", label=r"$M_{в}$")
# plt.plot(indexes_of_samples_first, values_of_samples_first, color="red", label=r"$M_{ср}$")
plt.scatter(indexes_of_samples_second, values_of_samples_second, color="blue", linewidth=3)
# plt.scatter(indexes_of_samples_first, values_of_samples_first, color="red", linewidth=3)

# # Add grid, title, axis labels and legend
plt.grid(True)
plt.title(
    "Зависимость разницы максимальных значений\nамплитуды от размера набора сэмплов",
    fontsize=14,
    linespacing=1,
)
plt.xlabel("Размер сэмплирования", fontsize=14, linespacing=1.5)
plt.ylabel("Различия громкости, Дб", fontsize=14, linespacing=1.5)
# plt.legend()

# Show the plot
plt.show()


#
