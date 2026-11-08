
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
import xgboost as xgb
import numpy as np
import os

# 1. 加载训练好的模型
model = xgb.XGBClassifier()
model.load_model('model_PVK.json')  # 加载之前保存的模型

# 2. 加载新数据进行预测
new_data_path = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/XRD_results/DPVK-OQMD-cif_no_bandgap0_Conventional_with_XRD_Fxrd_St改_numbered.csv'))

# 检查文件是否存在
if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"文件未找到: {new_data_path}")

# 加载新数据
new_data = pd.read_csv(new_data_path)

# 3. 复制原始数据以保留所有特征描述符
full_data_with_predictions = new_data.copy()

# 4. 准备用于预测的数据（删除目标列和不需要的列）
delect_col = ['entry_id','name','icsd_id','composition_generic','syngony','spacegroup',
                    'prototype','ntypes','natoms','delta_e','band_gap',
                    'stability','structure_id','xrd_status']

if 'name' in full_data_with_predictions.columns:
    # 如果name列存在，暂时保留它以便对齐结果
    identifier_col = ['name']
else:
    identifier_col = []
    full_data_with_predictions.insert(0, 'name', range(len(full_data_with_predictions)))

# 仅删除实际存在的列
cols_to_drop = [col for col in delect_col if col in full_data_with_predictions.columns]
X_predict = full_data_with_predictions.drop(cols_to_drop + identifier_col, axis=1, errors='ignore')

# 5. 进行预测
# 获取预测类别（0=非半导体，1=半导体）
predictions = model.predict(X_predict)
# 获取预测概率
probabilities = model.predict_proba(X_predict)[:, 1]

# 6. 将预测结果添加到原始数据中
full_data_with_predictions['predicted_class'] = predictions
full_data_with_predictions['probability_SEMI'] = probabilities
full_data_with_predictions['predicted_status'] = full_data_with_predictions['predicted_class'].map(
    {0: 'Non-PVK', 1: 'PVK'})

# 7. 保存包含预测结果的全部数据到新CSV文件
output_path = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/DPVK-OQMD_with_predictions.csv'))
full_data_with_predictions.to_csv(output_path, index=False)
print(f"预测完成! 结果已添加到数据表并保存到: {output_path}")

# 可选：在控制台打印部分结果预览
print("\n包含预测结果的数据预览:")
print(full_data_with_predictions[['name', 'predicted_status', 'probability_SEMI']].head())