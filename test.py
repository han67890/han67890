import networkx as nx
import matplotlib.pyplot as plt
import math

def offset_point(x1, y1, x2, y2, offset):
    """
    根据给定的 offset，将 (x1, y1) 沿着指向 (x2, y2) 的方向
    平移 offset 距离，返回平移后的坐标 (x_new, y_new)。
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    # 如果起点和终点重合，直接返回原点
    if dist == 0:
        return x1, y1
    # 计算单位方向向量，然后乘以 offset
    x_new = x1 + (dx / dist) * offset
    y_new = y1 + (dy / dist) * offset
    return x_new, y_new

# 1. 创建有向图
G = nx.DiGraph()

# 2. 定义节点
nodes = [
    "Radar Data Acquisition",
    "Signal Preprocessing\nGenerate Range-Doppler Map",
    "2D-CFAR Algorithm Filtering\nExtract Target Signal",
    "Micro-Doppler Effect\nExtract Human Signal",
    "MUSIC Algorithm\nAngle Estimation",
    "HDBSCAN Clustering\nSpatial Clustering",
    "Polynomial Regression Fitting\nOptimize Trajectory",
    "Target Localization Result"
]

# 3. 定义边
edges = [
    ("Radar Data Acquisition", "Signal Preprocessing\nGenerate Range-Doppler Map"),
    ("Signal Preprocessing\nGenerate Range-Doppler Map", "2D-CFAR Algorithm Filtering\nExtract Target Signal"),
    ("2D-CFAR Algorithm Filtering\nExtract Target Signal", "Micro-Doppler Effect\nExtract Human Signal"),
    ("Micro-Doppler Effect\nExtract Human Signal", "MUSIC Algorithm\nAngle Estimation"),
    ("MUSIC Algorithm\nAngle Estimation", "HDBSCAN Clustering\nSpatial Clustering"),
    ("HDBSCAN Clustering\nSpatial Clustering", "Polynomial Regression Fitting\nOptimize Trajectory"),
    ("Polynomial Regression Fitting\nOptimize Trajectory", "Target Localization Result")
]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# 4. 计算布局
pos = nx.spring_layout(G, k=1, seed=42)

# 5. (可选) 对布局进行缩放，让图中留出边距
pos = nx.rescale_layout_dict(pos, scale=1.5)

# 6. 设置画布大小
plt.figure(figsize=(12, 8))

# 7. 绘制节点
nx.draw_networkx_nodes(
    G, pos,
    node_color='lightblue',
    node_size=3000
)

# 8. 绘制节点标签
nx.draw_networkx_labels(
    G, pos,
    font_size=9,
    font_family='sans-serif',
    verticalalignment='center'
)

# 9. 在第一个节点上方标注 "START"
x_start, y_start = pos["Radar Data Acquisition"]
plt.text(
    x_start, y_start + 0.1,
    "START",
    fontsize=10,
    ha='center',
    va='bottom',
    color='red'
)

# 10. 手动绘制箭头，避免穿过节点圆
for (u, v) in edges:
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    # 根据节点大小和布局，需要自行微调 offset
    # 这里假设 offset=0.15 (或 0.2) 可以避免穿过圆
    offset_val = 0.15

    # 起点从 u 向 v 偏移 offset_val
    start_x, start_y = offset_point(x1, y1, x2, y2, offset_val)
    # 终点从 v 向 u 偏移 offset_val
    end_x, end_y = offset_point(x2, y2, x1, y1, offset_val)

    plt.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(
            arrowstyle='-|>',
            color='black',
            lw=2,
            connectionstyle='arc3,rad=0.2',
            # 不再使用 shrinkA/shrinkB，让偏移点本身决定箭头落点
            shrinkA=0,
            shrinkB=0
        )
    )

plt.title("Radar Data Processing Flowchart", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
