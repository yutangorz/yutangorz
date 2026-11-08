
# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('02_ALIGNN_TRAINING/src/training')))
import os
import csv
import shutil
import random
from pathlib import Path

# 设置随机种子以确保结果可复现
random.seed(42)

# 配置路径
source_folder = "xiong2"  # 源文件夹路径
id_prop_file = os.path.join(source_folder, "id_prop.csv")  # id_prop.csv文件路径
test_ratio = 0.1  # 测试集比例

# 创建输出文件夹
train_folder = os.path.join(source_folder, "train")
test_folder = os.path.join(source_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 读取id_prop.csv文件
print(f"从 {id_prop_file} 读取数据...")
file_ids = []  # 存储文件名
bandgaps = []  # 存储带隙值

with open(id_prop_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            file_ids.append(row[0].strip())
            bandgaps.append(row[1].strip())

print(f"共找到 {len(file_ids)} 个文件")

# 随机划分数据集
all_indices = list(range(len(file_ids)))
random.shuffle(all_indices)

test_size = int(len(file_ids) * test_ratio)
test_indices = set(all_indices[:test_size])
train_indices = set(all_indices[test_size:])

print(f"划分结果: 训练集 {len(train_indices)} 个文件, 测试集 {len(test_indices)} 个文件")

# 创建并写入训练集的id_prop.csv
train_id_prop = os.path.join(train_folder, "id_prop.csv")
with open(train_id_prop, "w", newline="") as f:
    writer = csv.writer(f)
    for idx in train_indices:
        writer.writerow([file_ids[idx], bandgaps[idx]])

print(f"已创建训练集id_prop.csv: {train_id_prop}")

# 创建并写入测试集的id_prop.csv
test_id_prop = os.path.join(test_folder, "id_prop.csv")
with open(test_id_prop, "w", newline="") as f:
    writer = csv.writer(f)
    for idx in test_indices:
        writer.writerow([file_ids[idx], bandgaps[idx]])

print(f"已创建测试集id_prop.csv: {test_id_prop}")

# 获取源文件夹中的所有文件（排除id_prop.csv）
all_files = []
for f in os.listdir(source_folder):
    if f != "id_prop.csv" and os.path.isfile(os.path.join(source_folder, f)):
        all_files.append(f)

print(f"找到 {len(all_files)} 个需要处理的文件")

# 复制文件到相应的文件夹
print("开始复制文件...")
copied_count = 0

# 创建一个从文件名到其在file_ids中索引的映射
file_to_index = {}
for idx, file_id in enumerate(file_ids):
    # 尝试不同的可能文件名格式
    possible_names = [
        file_id,
        f"{file_id}.vasp",
        f"{file_id}.cif",
        f"{file_id}.POSCAR",
        f"{file_id}.CONTCAR",
        f"POSCAR_{file_id}",
        f"CONTCAR_{file_id}"
    ]

    for name in possible_names:
        file_to_index[name] = idx

# 复制文件
for file_name in all_files:
    if file_name in file_to_index:
        idx = file_to_index[file_name]
        source_path = os.path.join(source_folder, file_name)

        if idx in train_indices:
            dest_path = os.path.join(train_folder, file_name)
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        elif idx in test_indices:
            dest_path = os.path.join(test_folder, file_name)
            shutil.copy2(source_path, dest_path)
            copied_count += 1
    else:
        print(f"警告: 文件 {file_name} 未在id_prop.csv中找到对应项")

print(f"成功复制 {copied_count} 个文件")

# 验证结果
train_files = [f for f in os.listdir(train_folder) if f != "id_prop.csv"]
test_files = [f for f in os.listdir(test_folder) if f != "id_prop.csv"]

print(f"\n验证结果:")
print(f"训练文件夹包含 {len(train_files)} 个VASP文件 + 1个id_prop.csv")
print(f"测试文件夹包含 {len(test_files)} 个VASP文件 + 1个id_prop.csv")

# 检查id_prop.csv中的记录数与实际文件数是否匹配
with open(train_id_prop, "r") as f:
    train_records = sum(1 for _ in csv.reader(f))

with open(test_id_prop, "r") as f:
    test_records = sum(1 for _ in csv.reader(f))

print(f"训练集id_prop.csv记录数: {train_records}")
print(f"测试集id_prop.csv记录数: {test_records}")

# 检查是否有文件未被处理
if copied_count == len(all_files):
    print("所有文件已成功处理!")
else:
    print(f"警告: 处理文件数量不匹配。原始文件: {len(all_files)}, 已复制: {copied_count}")

# 打印一些样本文件以帮助调试
print("\n样本文件:")
print(f"id_prop.csv中的前5个ID: {file_ids[:5]}")
print(f"源文件夹中的前5个文件: {all_files[:5]}")