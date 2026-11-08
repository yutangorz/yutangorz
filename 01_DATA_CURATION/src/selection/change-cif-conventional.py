
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
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import warnings
import traceback
import os

# 忽略pymatgen可能的警告
warnings.filterwarnings("ignore", module="pymatgen")


def convert_to_conventional_cif(cif_str):
    """将CIF字符串转换为传统晶胞的CIF字符串"""
    try:
        # 从字符串读取结构
        structure = Structure.from_str(cif_str, fmt="cif")

        # 分析空间群并获取传统晶胞
        analyzer = SpacegroupAnalyzer(structure)
        conventional_structure = analyzer.get_conventional_standard_structure()

        # 直接将结构转换为CIF字符串
        return str(CifWriter(conventional_structure))

    except Exception as e:
        # 记录错误信息但继续处理
        print(f"转换失败: {str(e)}")
        print(traceback.format_exc())
        return cif_str  # 返回原始字符串保持数据完整


# 主处理流程
def process_cif_file(input_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 创建新列存储转换结果
    conventional_cifs = []

    # 遍历并转换每个CIF字符串
    print(f"开始处理 {len(df)} 个CIF结构...")
    for i, cif_str in enumerate(df["cif"]):
        if i % 100 == 0:
            print(f"处理进度: {i + 1}/{len(df)}")
        conventional_cifs.append(convert_to_conventional_cif(cif_str))

    # 更新DataFrame中的cif列
    df["cif"] = conventional_cifs

    # 生成输出文件名
    output_file = input_file.replace(".csv", "_Conventional.csv")

    # 保存处理后的数据
    df.to_csv(output_file, index=False)
    print(f"\n处理完成! 已保存到: {output_file}")
    return output_file


# 文件路径配置
input_csv = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/DPVK-OQMD-cif_no_bandgap0.csv'))

# 执行处理
if __name__ == "__main__":
    output_path = process_cif_file(input_csv)
    print(f"新文件已创建: {output_path}")