import os
import json
import pandas as pd

# 输入路径
results_csv = "/home/phy/lab/Safebound/SafeBound/res/aids_merged/results.csv"
base_json_dir = "/home/phy/lab/graph_card/graph-card-est-2024/estimation/aids_merged_safebound"

# 读取 CSV 文件
df = pd.read_csv(results_csv)

# 确保列存在
assert "folder" in df.columns and "query_name" in df.columns and "cardinality_bound" in df.columns, "CSV 文件缺少必要列"

# 遍历每一行
for index, row in df.iterrows():
    folder = row["folder"]
    query_name = row["query_name"]
    bound = row["cardinality_bound"]

    # 构造 JSON 文件路径
    json_dir = os.path.join(base_json_dir, folder)
    json_path = os.path.join(json_dir, f"{query_name}.json")

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        continue

    # 读取 JSON 文件
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"JSON 解码失败: {json_path}")
        continue

    # 修改 count 字段
    if "count" in data:
        data["count"] = bound
        # print(f"已更新 {json_path} 中的 count 为 {bound}")
    else:
        print(f"{json_path} 中无 count 字段，跳过更新")
        continue

    # 写回 JSON 文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)