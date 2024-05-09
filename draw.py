import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df1 = pd.read_csv('Data_Csv/Fcn-Evaluate_mIoU.csv')
df2 = pd.read_csv('Data_Csv/Unet-Evaluate_mIoU.csv')
# df3 = pd.read_csv('data3.csv')

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制曲线
ax.plot(df1['x'], df1['y'], label='Data 1')
ax.plot(df2['x'], df2['y'], label='Data 2')
# ax.plot(df3['x'], df3['y'], label='Data 3')

# 添加图例和标题
ax.legend()
ax.set_title('Multiple CSV Data Visualization')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 显示图形
plt.show()