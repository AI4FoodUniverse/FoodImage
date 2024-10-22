import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Read the CSV file
# 读取CSV文件
data = pd.read_csv('label/4cbatch1label.csv')

# Extract the data from the 'Firmness' column
# 提取Firmness列的数据
firmness_data = data['Firmness']

# Plot the histogram
# 绘制直方图
plt.hist(firmness_data, bins=20, density=True, alpha=0.6, color='b')

# Fit a normal distribution curve
# 绘制正态分布曲线
mu, sigma = stats.norm.fit(firmness_data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

# Add labels
# 添加标签
plt.xlabel('Firmness')
plt.ylabel('Frequency')
plt.title('Histogram of Firmness with Normal Distribution Fit')
plt.grid(True)

# Show the plot
# 显示图形
plt.show()

# Print the parameters of the fitted normal distribution
# 打印正态分布的拟合参数
print(f"Fitted normal distribution parameters: Mean={mu}, Standard Deviation={sigma}")