
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
from pymatgen.core import Structure
import pandas as pd
import numpy as np
import re
from math import gcd
from functools import reduce
import traceback
import os
import time
import sys

input_file = str(_repo_path('external_inputs/original_workspace/Feature enginer/test.csv'))
base_name = os.path.splitext(input_file)[0]
output_file = f"{base_name}_St.csv"

# =================== 辅助函数模块 ===================
def vector_angle(v1, v2):
    """计算两个向量之间的夹角（单位：度）"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))


def parse_formula(cif_content):
    """从CIF内容解析化学式"""
    for line in cif_content.split('\n'):
        if "_chemical_formula_sum" in line:
            formula_str = re.findall(r"'(.+)'", line)[0]
            return re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)
    raise ValueError("未找到化学式信息")


def get_simplified_ratio(element_counts):
    """获取元素的最简整数比"""
    counts = list(element_counts.values())
    gcd_val = reduce(lambda x, y: gcd(x, y), counts)
    return {k: v // gcd_val for k, v in element_counts.items()}


def determine_structure_type(simplified):
    """改进的结构类型判断"""
    counts = sorted(simplified.values(), reverse=True)
    if counts == [6, 2, 1, 1]:
        return "double"
    elif counts == [3, 1, 1]:
        return "single"
    raise ValueError("无法识别的结构类型")


def calc_min_distance(struct, from_indices, to_indices):
    """计算原子间最小距离"""
    if not from_indices or not to_indices:
        return np.nan
    distances = []
    for i in from_indices:
        valid = [j for j in to_indices if j != i]
        if not valid:
            continue
        distances.append(min(struct.get_distance(i, j) for j in valid))
    return np.nanmean(distances) if distances else np.nan


def calc_angle(struct, center_indices, target1_indices, target2_indices):
    """计算原子间角度"""
    angles = []
    for i in center_indices:
        try:
            site = struct[i]
            t1_list = [j for j in target1_indices if j != i]
            t2_list = [j for j in target2_indices if j != i]

            if not t1_list or not t2_list:
                continue

            t1 = min(t1_list, key=lambda j: site.distance(struct[j]))
            t2 = min(t2_list, key=lambda j: site.distance(struct[j]))

            vec1 = struct[t1].coords - site.coords
            vec2 = struct[t2].coords - site.coords

            if np.linalg.norm(vec1) < 1e-6 or np.linalg.norm(vec2) < 1e-6:
                continue

            angles.append(vector_angle(vec1, vec2))
        except:
            continue
    return np.nanmean(angles) if angles else np.nan


# =================== CIF处理函数 ===================
def process_cif(cif_content):

    try:
        struct = Structure.from_str(cif_content, fmt="cif")
        struct.merge_sites(tol=0.01)

        formula_elements = parse_formula(cif_content)
        element_counts = {}
        for elem, cnt in formula_elements:
            cnt = int(cnt) if cnt else 1
            element_counts[elem] = element_counts.get(elem, 0) + cnt
        simplified = get_simplified_ratio(element_counts)
        struct_type = determine_structure_type(simplified)

        sorted_elements = sorted(simplified.items(), key=lambda x: x[1], reverse=True)

        if struct_type == "double":
            ele_x = next(e[0] for e in sorted_elements if e[1] == 6)
            ele_a = next(e[0] for e in sorted_elements if e[1] == 2)
            remaining = [e[0] for e in sorted_elements if e[0] not in (ele_x, ele_a)]
            ele_b1, ele_b2 = remaining[0], remaining[1] if len(remaining) >= 2 else remaining[0]

        else:
            ele_x = next(e[0] for e in sorted_elements if e[1] == 3)
            remaining = [e[0] for e in sorted_elements if e[0] not in (ele_x)]
            ele_a, ele_b1 = remaining[0], remaining[1] if remaining else None
            ele_b2 = ele_b1


        ele_x_indices = [i for i, s in enumerate(struct) if s.specie.name == ele_x]
        ele_a_indices = [i for i, s in enumerate(struct) if s.specie.name == ele_a]
        ele_b1_indices = [i for i, s in enumerate(struct) if s.specie.name == ele_b1]
        ele_b2_indices = ([i for i, s in enumerate(struct) if s.specie.name == ele_b2]
                          if struct_type == "double" else ele_b1_indices)

        descriptors = {
            'a': struct.lattice.a,
            'b': struct.lattice.b,
            'c': struct.lattice.c,
            'alpha': struct.lattice.alpha,
            'beta': struct.lattice.beta,
            'gamma': struct.lattice.gamma,
            'volume': struct.volume,
            'r_AB1': calc_min_distance(struct, ele_a_indices, ele_b1_indices),
            'r_AX': calc_min_distance(struct, ele_a_indices, ele_x_indices),
            'r_B1X': calc_min_distance(struct, ele_b1_indices, ele_x_indices),
            'struct_type': struct_type,
            'θ_AB1X': np.nan, 'θ_AB1A': np.nan, 'θ_XB1X': np.nan,
            'θ_B1AX': np.nan, 'θ_B1AB1': np.nan, 'θ_XAX': np.nan,
            'θ_B1XA': np.nan, 'θ_B1XB1': np.nan, 'θ_AXA': np.nan,
            'θ_AB2X': np.nan, 'θ_AB2A': np.nan, 'θ_XB2X': np.nan,
            'θ_B2AX': np.nan, 'θ_B2AB2': np.nan, 'θ_B2XA': np.nan,
            'θ_B2XB2': np.nan
        }

        descriptors['θ_AB1X'] = calc_angle(struct, ele_b1_indices, ele_a_indices, ele_x_indices)
        descriptors['θ_AB1A'] = calc_angle(struct, ele_b1_indices, ele_a_indices, ele_a_indices)
        descriptors['θ_XB1X'] = calc_angle(struct, ele_b1_indices, ele_x_indices, ele_x_indices)
        descriptors['θ_B1AX'] = calc_angle(struct, ele_a_indices, ele_b1_indices, ele_x_indices)
        descriptors['θ_B1AB1'] = calc_angle(struct, ele_b1_indices, ele_a_indices, ele_b1_indices)
        descriptors['θ_XAX'] = calc_angle(struct, ele_x_indices, ele_a_indices, ele_x_indices)
        descriptors['θ_B1XA'] = calc_angle(struct, ele_x_indices, ele_b1_indices, ele_a_indices)
        descriptors['θ_B1XB1'] = calc_angle(struct, ele_b1_indices, ele_x_indices, ele_b1_indices)
        descriptors['θ_AXA'] = calc_angle(struct, ele_a_indices, ele_x_indices, ele_a_indices)

        if struct_type == "double":
            descriptors['r_AB2'] = calc_min_distance(struct, ele_a_indices, ele_b2_indices)
            descriptors['r_B2X'] = calc_min_distance(struct, ele_b2_indices, ele_x_indices)
            descriptors['θ_AB2X'] = calc_angle(struct, ele_b2_indices, ele_a_indices, ele_x_indices)
            descriptors['θ_AB2A'] = calc_angle(struct, ele_b2_indices, ele_a_indices, ele_a_indices)
            descriptors['θ_XB2X'] = calc_angle(struct, ele_b2_indices, ele_x_indices, ele_x_indices)
            descriptors['θ_B2AX'] = calc_angle(struct, ele_a_indices, ele_b2_indices, ele_x_indices)
            descriptors['θ_B2AB2'] = calc_angle(struct, ele_b2_indices, ele_a_indices, ele_b2_indices)
            descriptors['θ_B2XA'] = calc_angle(struct, ele_x_indices, ele_b2_indices, ele_a_indices)
            descriptors['θ_B2XB2'] = calc_angle(struct, ele_b2_indices, ele_x_indices, ele_b2_indices)
        else:  # 单钙钛矿时复制B1参数到B2
            descriptors['r_AB2'] = descriptors['r_AB1']
            descriptors['r_B2X'] = descriptors['r_B1X']
            descriptors['θ_AB2X'] = descriptors['θ_AB1X']
            descriptors['θ_AB2A'] = descriptors['θ_AB1A']
            descriptors['θ_XB2X'] = descriptors['θ_XB1X']
            descriptors['θ_B2AX'] = descriptors['θ_B1AX']
            descriptors['θ_B2AB2'] = descriptors['θ_B1AB1']
            descriptors['θ_B2XA'] = descriptors['θ_B1XA']
            descriptors['θ_B2XB2'] = descriptors['θ_B1XB1']

        return descriptors

    except Exception as e:
        print(f"处理CIF时发生错误: {str(e)}")
        print(traceback.format_exc())
        return None


# =================== 主程序 ===================
if __name__ == "__main__":

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：输入文件 {input_file} 未找到")
        exit(1)

    total = len(df)
    print(f"开始处理 {total} 条记录...")

    results = []
    start_time = time.time()
    for idx, row in df.iterrows():
        current = idx + 1
        try:
            progress = current / total * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / current if current > 0 else 0
            remaining = (total - current) * avg_time

            sys.stdout.write(f"\r▶ 进度: {current}/{total} ({progress:.1f}%) | "
                             f"已用: {elapsed:.0f}s | 剩余: {remaining:.0f}s | "
                             f"当前处理: {row['name']}")
            sys.stdout.flush()

            descriptors = process_cif(row['cif'])
            results.append(descriptors if descriptors else {})

        except Exception as e:
            print(f"\n⚠ 处理 {row['name']} 时发生错误: {str(e)}")
            results.append({})

    print(f"\n✅ 处理完成！总耗时: {time.time() - start_time:.1f}秒")

    descriptors_df = pd.DataFrame(results)
    final_df = pd.concat([df[['name', 'delta_e', 'stability', 'band_gap', 'spacegroup', 'cif', 'poscar']],
                              descriptors_df], axis=1)

    final_df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")