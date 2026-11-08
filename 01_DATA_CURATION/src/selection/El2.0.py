
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
import re
import numpy as np
from collections import defaultdict


def parse_formula(formula):
    """解析化学式，返回元素及其原子个数的字典"""
    elements_count = {}
    # 使用正则表达式匹配元素及其后的数字
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)

    for element, count_str in matches:
        count = 1 if count_str == '' else int(count_str)
        elements_count[element] = elements_count.get(element, 0) + count

    return elements_count


def get_electron_affinity_data():
    """获取元素的电子亲和势数据"""
    return {
        'H': 72.77, 'He': -48, 'Li': 59.63, 'Be': -240, 'B': 26.99,
        'C': 153.89, 'N': -7, 'O': 140.98, 'F': 328.16, 'Ne': -116,
        'Na': 52.87, 'Mg': -39.6, 'Al': 41.76, 'Si': 133.6, 'P': 72.04,
        'S': 200.4, 'Cl': 349, 'Ar': -96, 'K': 48.38, 'Ca': -156,
        'Sc': 18.1, 'Ti': 7.6, 'V': 50.91, 'Cr': 64.3, 'Mn': -50,
        'Fe': 15.7, 'Co': 63.7, 'Ni': -111.5, 'Cu': 119.2, 'Zn': -58,
        'Ga': 41.1, 'Ge': 119, 'As': 77.6, 'Se': 194.96, 'Br': 324.7,
        'Kr': -96, 'Rb': 46.88, 'Sr': -39.1, 'Y': 29.6, 'Zr': 41.1,
        'Nb': 88.6, 'Mo': 72.1, 'Tc': 53, 'Ru': 101, 'Rh': 109,
        'Pd': 54, 'Ag': 125.86, 'Cd': -68, 'In': 43, 'Sn': 115,
        'Sb': 101, 'Te': 190.2, 'I': 295.3, 'Xe': -77, 'Cs': 45.5,
        'Ba': -39.9, 'La': 48, 'Ce': 57, 'Pr': 93, 'Nd': 50,
        'Pm': 41, 'Sm': 15.63, 'Eu': -55, 'Gd': 99.9, 'Tb': 92.3,
        'Dy': 39.3, 'Ho': 50, 'Er': 32.6, 'Tm': 1.029, 'Yb': -1.93,
        'Lu': -1.05, 'Hf': 17.18, 'Ta': 43.6, 'W': 78.76, 'Re': 126,
        'Os': 101.1, 'Ir': 151, 'Pt': 205, 'Au': 222.8, 'Hg': -48,
        'Tl': 36.4, 'Pb': 50, 'Bi': 91, 'Po': 136, 'At': 233,
        'Rn': -77, 'Fr': 44, 'Ra': -44, 'Ac': 37, 'Th': 112,
        'Pa': 53, 'U': 50.4, 'Np': 45.6, 'Pu': -48, 'Am': 41,
        'Cm': 39, 'Bk': 37, 'Cf': 36, 'Es': 35, 'Fm': 34,
    }


# 主程序
def main():
    # 1. 读取原始数据
    input_path = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/Date_with_feature/DPVK-OQMD_no_bandgap0_pvk_st_Fxrd_MP_less.csv'))
    output_path = input_path.replace('.csv', '_El2.0.csv')

    df = pd.read_csv(input_path)

    # 2. 获取电子亲和势数据
    ea_data = get_electron_affinity_data()

    # 3. 为新的描述符创建空列表
    min_ea = []  # 最小值
    max_ea = []  # 最大值
    range_ea = []  # 范围
    wmean_ea = []  # 加权平均值（考虑原子个数）
    wavg_dev_ea = []  # 加权平均绝对偏差（考虑原子个数）

    # 4. 对每行数据进行处理
    for formula in df['name']:
        # 解析化学式，获取元素及其个数
        elements_count = parse_formula(formula)

        # 获取每个元素的电子亲和势值
        ea_values = []
        atom_counts = []

        for element, count in elements_count.items():
            if element in ea_data:
                affinity = ea_data[element]
                # 每个原子对应一个亲和势值
                ea_values.extend([affinity] * count)
                atom_counts.extend([count] * count)

        # 计算统计数据
        if ea_values:  # 确保有值
            ea_array = np.array(ea_values)

            # 简单统计量（不考虑权重）
            min_ea.append(np.min(ea_array))
            max_ea.append(np.max(ea_array))
            range_ea.append(np.max(ea_array) - np.min(ea_array))

            # 计算加权平均值（考虑原子个数）
            weighted_mean = np.average(ea_array, weights=atom_counts)
            wmean_ea.append(weighted_mean)

            # 计算加权平均绝对偏差
            abs_deviations = np.abs(ea_array - weighted_mean)
            wavg_dev_ea.append(np.average(abs_deviations, weights=atom_counts))
        else:
            # 如果没有有效值，填充NaN
            min_ea.append(np.nan)
            max_ea.append(np.nan)
            range_ea.append(np.nan)
            wmean_ea.append(np.nan)
            wavg_dev_ea.append(np.nan)

    # 5. 将计算出的新描述符添加到DataFrame
    df['min_ElAffinity'] = min_ea
    df['max_ElAffinity'] = max_ea
    df['range_ElAffinity'] = range_ea
    df['wmean_ElAffinity'] = wmean_ea
    df['wavg_dev_ElAffinity'] = wavg_dev_ea

    # 6. 保存为新的CSV文件
    df.to_csv(output_path, index=False)
    print(f"处理完成，新文件已保存至: {output_path}")
    print(f"原始数据行数: {len(df)}, 新添加列数: 5")
    print(f"新描述符示例：")
    print(df[['name', 'min_ElAffinity', 'wmean_ElAffinity']].head(3))


if __name__ == "__main__":
    main()