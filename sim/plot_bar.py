import matplotlib.pyplot as plt
import numpy as np
import os

categories = ["64MB", "96MB", "128MB", "192MB", "0.75GB", "1.5GB"]
real_system_4GPUs = [
    816.3491943844161,
    1236.765004973329,
    1685.4075988565194,
    2423.5586865981572,
    9795.868267882179,
    19567.65817551643,
]

real_system_16GPUs = [
    1151.0201144844164,
    1657.263348617751,
    2188.684262744053,
    3210.104250860299,
    12535.39779180638,
    24727.523917784394,
]
palm_simulator_4GPUs = [806, 1206, 1606, 2406, 9606, 19206]

palm_simulator_16GPUs = [1120, 1620, 2120, 3120, 12120, 24120]

plt.rcParams["font.weight"] = "bold"

# 创建两个子图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# Set default font weight to bold

# 子图1：4 GPUs
bar_width = 0.35
bar_positions_4_GPUs = np.arange(len(categories))
axes[0].bar(bar_positions_4_GPUs, real_system_4GPUs, bar_width, label="Astra-sim2.0")
axes[0].bar(
    bar_positions_4_GPUs + bar_width,
    palm_simulator_4GPUs,
    bar_width,
    label="PALM ",
)
axes[0].set_title("4 GPUs")
axes[0].set_ylabel("Running TIme (\u03BCs)")
axes[0].set_xticks(bar_positions_4_GPUs + bar_width / 2)
axes[0].set_xticklabels(categories)
axes[0].set_xlabel("Collective Size")
axes[0].legend()

# 子图2：16 GPUs
bar_positions_16_GPUs = bar_positions_4_GPUs
axes[1].bar(bar_positions_16_GPUs, real_system_16GPUs, bar_width, label="Astra-sim2.0")
axes[1].bar(
    bar_positions_16_GPUs + bar_width,
    palm_simulator_16GPUs,
    bar_width,
    label="PALM ",
)
axes[1].set_title("16 GPUs")
# axes[1].set_ylabel("Performance Metric")
axes[1].set_xticks(bar_positions_16_GPUs + bar_width / 2)
axes[1].set_xticklabels(categories)
axes[1].set_xlabel("Collective Size")
axes[1].legend()

for ax in axes:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

# 设置全局标题
# fig.suptitle("Comparison of Real System and PALM Simulator", fontsize=16)

# plt.title("Sample Bar Chart", fontdict={"fontname": "Arial", "fontsize": 16})
# plt.xlabel("Collective Size", fontdict={"fontname": "Arial", "fontsize": 12})

# 创建保存图像的文件夹（如果不存在）
output_folder = "./figs"
os.makedirs(output_folder, exist_ok=True)

# 保存图像到文件夹中
output_file_path = os.path.join(output_folder, "example_plot.pdf")
plt.savefig(output_file_path, format="pdf")

# 显示图形
plt.show()

# print(f"real_system_4GPUs = {real_system_4GPUs}")

# print(f"type of real_system_4GPUs = {type(real_system_4GPUs)}")
