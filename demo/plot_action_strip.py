import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

mapping_file = '/home/fitz_joye/TSM-action-recognition/data/assembly101/assembly101-annotations/fine-grained-annotations/actions.csv'
mapping_file2 = '/home/fitz_joye/TSM-action-recognition/data/assembly101/assembly101-annotations/coarse-annotations/actions.csv'
with open(mapping_file2) as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines[1:]] # 跳过标题行
# lines = [line.split(',')[4] for line in lines] # 只保留动作名称
lines = [line.split(',')[3] for line in lines] # 只保留动作名称
actions_dict = {}
reverse_actions_dict = {}
for i, action in enumerate(lines):
    actions_dict[action] = i
    reverse_actions_dict[i] = action



def plot_action_segmentation_strip(series, re_action_dict=reverse_actions_dict):
    series = [actions_dict[action] for action in series]  # 将动作名称转换为索引
    n_frames = len(series)  # 获取时间序列的长度
    action_matrix = np.zeros((1, n_frames))
    action_matrix[0] = series  # 将序列填充进矩阵

    # 使用 Spectral 调色板
    cmap = cm.Spectral  
    fig, ax = plt.subplots(figsize=(10, 6))  # 设置图像大小

    # 绘制条带图
    cax = ax.imshow(action_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    # 设置轴标签
    ax.set_xlabel("Time (#Frames)")
    ax.set_yticks([])  # 去除 Y 轴标签

    # 显示颜色条
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('Action Categories')  # 给颜色条添加标签

    # 提取唯一的动作类别
    unique_actions = np.unique(series)
    
    # 如果没有传入 actions_dict，则使用自动生成的类别映射
    if re_action_dict is None:
        re_action_dict = {i: f'Action {i}' for i in unique_actions}
    
    # 为每个动作类别分配颜色
    colors = [cmap(i / len(unique_actions)) for i in range(len(unique_actions))]
    
    # 获取每个动作类别的标签
    labels = [re_action_dict[action] for action in unique_actions]

    # 为每个类别创建图例
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize='small')

    # 布局调整，避免重叠
    plt.tight_layout()

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # 转换为RGB格式
    image_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)

# 提取 RGB 数据（去除 alpha 通道）
    image_rgb = image_argb[:, :, 1:]  # 选择 ARGB 中的 RGB 部分（即从索引 1 到 3）
    return image_rgb