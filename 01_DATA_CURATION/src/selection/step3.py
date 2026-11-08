
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
_repository_sys.path.insert(0, str(_repo_path('01_DATA_CURATION/src/selection')))
import pandas as pd

# 完整空间群符号-编号映射表（包含所有缺失项）
spacegroup_mapping = {
    # 立方晶系
    "Fm-3m": 225, "Pm-3m": 221, "Fd-3m": 227, "Im-3": 204, "Pm-3n": 223,"P213": 198,"Pa-3": 205,
    # 四方晶系
    "I4/m": 87, "I4/mmm": 139, "P4mm": 99, "I4/mcm": 140, "P4/mmm": 123,
    "P4/mbm": 127, "I4mm": 107, "P4/nmm": 129, "P4/mnc": 128, "P42/nmc": 137,
    "P42/mnm": 136, "P4bm": 100, "I-4m2": 119, "P4/nbm": 125, "P42/n": 86,
    "P-421m": 113, "P42/m": 84, "P42": 77, "P42mc": 105,"P41": 76,"P-421c": 114,
    # 三方晶系
    "R-3": 148, "R-3c": 167, "R3": 146, "R3m": 160, "R3c": 161,
    "P-3m1": 164, "P-31c": 163, "P-3c1": 165, "R32": 155, "P3m1": 156,
    "P-31m": 157, "P3121": 152, "P-3": 143, "P321": 150, "P32": 145,
    "P31c": 159, "P312": 149,"R-3m": 166,
    # 正交晶系
    "Pnma": 62, "Cmcm": 63, "Cmmm": 65, "Imma": 74, "Pbcm": 57,
    "Amm2": 38, "Ima2": 45, "Pmmm": 47, "Fmmm": 69, "Pmmn": 59,
    "Pbca": 61, "Pbam": 55, "Pnna": 52, "Ibam": 72, "Cmca": 64,
    "Pna21": 33, "Pnn2": 34, "Pcca": 49, "Imm2": 44, "Pba2": 32,
    "Cmc21": 36, "C2221": 20, "P212121": 19, "Pmc21": 26, "Pmm2": 25,
    "Pmmm": 47, "Immm": 71, "Ama2": 40, "Pmma": 51, "Pcca": 54,
    "Pbcn": 60, "Cmm2": 35, "Pmna": 53, "Pca21": 29, "Pccn": 56,
    "Pnc2": 30, "Cmma": 67,"Pmn21": 31,"Fmm2": 42,
    # 单斜晶系
    "P21/c": 14, "C2/m": 12, "C2/c": 15, "P21/m": 11, "P2/c": 13,
    "Pc": 7, "Cm": 8, "Cc": 9, "P2/m": 10, "P21": 4, "Pm": 6,"C2": 5,
    # 六方晶系
    "P63/mmc": 194, "P-6m2": 187,
    'P-1': 2,
    'P1': 1,
    'I41/a': 88,
    'P63mc': 186,
    'I-42d': 122,
    'P63cm': 185,
    'P63': 173,
    'P63/mcm': 193,
    'P31': 144,
    'Pn-3': 201,
    'P6522': 179,
    'P61': 169
}

syngony_mapping = {
    "Cubic": 1,
    "Tetragonal": 2,
    "Trigonal": 3,
    "Orthorhombic": 4,
    "Monoclinic": 5,
}

struct_type_mapping = {
    "single": 1,
    "double": 2,
}

# 读取数据集
file_path = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/XRD_results/DPVK-OQMD-cif_no_bandgap0_Conventional_with_XRD_Fxrd_St改.csv'))
df = pd.read_csv(file_path)

# 1. 找出并输出所有未映射的空间群符号（包括NaN）
unmapped_symbols = []
for symbol in df["spacegroup"].unique():
    if symbol not in spacegroup_mapping and pd.notna(symbol):
        unmapped_symbols.append(symbol)

if unmapped_symbols:
    print(f"警告：以下空间群符号未找到对应编号，请补充映射：")
    for symbol in unmapped_symbols:
        print(f"  - {symbol}")

# 2. 转换spacegroup列为编号
df["spacegroup"] = df["spacegroup"].map(spacegroup_mapping)
##df["syngony"] = df["syngony"].map(syngony_mapping)
##df["struct_type"] = df["struct_type"].map(struct_type_mapping)

# 3. 找出并显示含有NaN值的行
nan_rows = df[df["spacegroup"].isna()]
if not nan_rows.empty:
    print(f"\n警告：数据集中有 {len(nan_rows)} 行空间群信息缺失:")
    print(nan_rows[["spacegroup", "struct_type", "syngony"]].head())  # 显示部分缺失数据

# 4. 将缺失值填充为-1（异常标记）
df["spacegroup"] = df["spacegroup"].fillna(-1)

# 5. 保存修复后的数据集
output_path = file_path.replace(".csv", "_numbered.csv")
df.to_csv(output_path, index=False)
print(f"\n转换完成，结果已保存至：{output_path}")
print(f"总行数: {len(df)}, 缺失空间群信息行数: {len(nan_rows)}")