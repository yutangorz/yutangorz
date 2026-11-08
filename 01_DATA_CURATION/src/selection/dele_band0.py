
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

# 设置文件路径
input_path = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/DPVK-OQMD-cif.csv'))
output_path = input_path.replace(".csv", "_no_bandgap0.csv")  # 新文件名添加后缀

try:
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 检查band_gap列是否存在
    if "band_gap" not in df.columns:
        raise KeyError(f"数据集中未找到 'band_gap' 列。现有列名：{list(df.columns)}")

    # 计算删除前的行数和0值数量
    original_count = len(df)
    zero_bandgap_count = (df["band_gap"] == 0).sum()

    # 删除band_gap为0的行
    df_clean = df[df["band_gap"] != 0].copy()

    # 检查是否删除了所有数据
    if len(df_clean) == 0:
        print("警告：删除后数据集为空！请检查band_gap列的值范围")
    else:
        # 保存新数据集
        df_clean.to_csv(output_path, index=False)

        # 计算统计信息
        new_count = len(df_clean)
        removed_count = original_count - new_count
        removed_percent = (removed_count / original_count) * 100

        # 打印结果报告
        print(f"处理完成：已删除 {removed_count} 行（占总数据的 {removed_percent:.2f}%）")
        print(f"原始数据行数: {original_count}")
        print(f"剩余数据行数: {new_count}")
        print(f"删除的0值行数: {zero_bandgap_count}（可能由于数据转换存在小数）")
        print(f"新数据集已保存至: {output_path}")

except FileNotFoundError:
    print(f"错误：文件 '{input_path}' 未找到，请检查路径是否正确")
except KeyError as e:
    print(str(e))
except Exception as e:
    print(f"处理过程中发生意外错误: {str(e)}")