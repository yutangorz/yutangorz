
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
import numpy as np
from pathlib import Path


def find_peaks(intensities, angles):
    """识别XRD数据中的特征峰（局部极大值）"""
    peaks = []
    for i in range(1, len(intensities) - 1):
        if intensities[i] > intensities[i - 1] and intensities[i] > intensities[i + 1]:
            peaks.append((angles[i], intensities[i]))
    return peaks


def extract_features(peaks):
    """从识别的峰中提取所需特征"""
    features = {}

    # 特征峰数量
    features["num_peaks"] = len(peaks)

    # 按强度排序的峰
    intensity_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)

    # 前六个最强峰
    for i in range(1, 7):
        if i <= len(intensity_sorted):
            features[f"peak{i}_intensity"] = intensity_sorted[i - 1][1]
            features[f"peak{i}_angle"] = intensity_sorted[i - 1][0]
        else:
            features[f"peak{i}_intensity"] = 0
            features[f"peak{i}_angle"] = 0

    # 按角度排序的峰
    angle_sorted = sorted(peaks, key=lambda x: x[0])

    # 前六个最小角度的峰
    for i in range(1, 7):
        if i <= len(angle_sorted):
            features[f"min_angle_peak{i}"] = angle_sorted[i - 1][0]
        else:
            features[f"min_angle_peak{i}"] = 0

    # 前十强峰的统计特征
    top10 = intensity_sorted[:10] if len(intensity_sorted) > 10 else intensity_sorted

    if top10:
        top10_angles = [p[0] for p in top10]
        top10_intensities = [p[1] for p in top10]

        features["top10_angle_mean"] = np.mean(top10_angles)
        features["top10_intensity_mean"] = np.mean(top10_intensities)
        features["top10_angle_std"] = np.std(top10_angles)
        features["top10_intensity_std"] = np.std(top10_intensities)
    else:
        for stat in ["top10_angle_mean", "top10_intensity_mean",
                     "top10_angle_std", "top10_intensity_std"]:
            features[stat] = 0

    return features


# 文件路径
input_path = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/XRD_results/DPVK-OQMD-cif_processed_Conventional_St_with_XRD.csv'))
output_path = input_path.replace(".csv", "_Fxrd.csv")

# 读取数据
df = pd.read_csv(input_path)

# 获取角度列名
angle_columns = df.columns[37:]
angles = np.array([float(col.replace('°', '')) for col in angle_columns])

# 创建新列
new_columns = ["num_peaks"]
for i in range(1, 7):
    new_columns += [f"peak{i}_intensity", f"peak{i}_angle"]
for i in range(1, 7):
    new_columns += [f"min_angle_peak{i}"]
new_columns += ["top10_angle_mean", "top10_intensity_mean",
                "top10_angle_std", "top10_intensity_std"]

for col in new_columns:
    df[col] = np.nan

# 处理每一行的XRD数据
for idx, row in df.iterrows():
    xrd_data = row[angle_columns].astype(float).values
    peaks = find_peaks(xrd_data, angles)
    features = extract_features(peaks)

    for feature, value in features.items():
        df.loc[idx, feature] = value

# 删除原始XRD列
df.drop(columns=angle_columns, inplace=True)

# 保存处理后的数据
df.to_csv(output_path, index=False)
print(f"处理完成! 结果已保存至: {output_path}")
print(f"新增特征数量: {len(new_columns)}")
print(f"处理后数据维度: {df.shape}")