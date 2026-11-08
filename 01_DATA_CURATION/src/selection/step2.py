
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
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
import os

input_file = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/Date_with_feature/DPVK-OQMD_no_bandgap0_pvk_st_Fxrd.csv'))
base_name = os.path.splitext(input_file)[0]
output_file = f"{base_name}_MP.csv"

if __name__ == "__main__":  # 关键修改
    # 读取CSV文件（假设包含"formula"列）
    df = pd.read_csv(input_file)

    # Step 1: 将化学式字符串转换为composition对象[8](@ref)
    data = StrToComposition().featurize_dataframe(df, "name")

    # Step 2: 生成Magpie特征描述符[5](@ref)
    ep_feat = ElementProperty.from_preset("magpie")
    data = ep_feat.featurize_dataframe(data, col_id="composition")

    # Step 3: 清理数据列
    data = data.drop(columns=["composition"])  # 删除中间生成的composition列
    data.columns = data.columns.str.replace("MagpieData ", "")  # 简化列名

    # Step 4: 保存结果到新CSV文件[8](@ref)
    data.to_csv(output_file, index=False)