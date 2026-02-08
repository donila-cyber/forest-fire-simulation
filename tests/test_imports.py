import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
import scipy

print("✅ NumPy version:", np.__version__)
print("✅ Matplotlib version:", plt.matplotlib.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ SciPy version:", scipy.__version__)

# Создаем простой массив
arr = np.array([1, 2, 3, 4, 5])
print("✅ NumPy array:", arr)

# Создаем простой график
plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("Test plot")
plt.savefig("test_plot.png")
print("✅ Plot saved as test_plot.png")

# Создаем DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("✅ DataFrame created")
print(df)

print("\n🎉 All imports successful!")
