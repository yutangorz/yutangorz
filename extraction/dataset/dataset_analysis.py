import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# ================= 配置区域 =================
INPUT_FOLDER = "."
OUTPUT_EXCEL = "bx_top10_data.xlsx"
MIN_ANGLE_CUTOFF = 160.0  # <--- 核心修改：只保留大于这个角度的数据


# ============================================

def identify_B_and_X(structure):
    species_count = structure.composition.get_el_amt_dict()
    sorted_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_species) < 2: return None, None

    elem_x = sorted_species[0][0]
    elem_b = None
    alkali_earth = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']

    for elem, _ in sorted_species[1:]:
        if elem not in alkali_earth:
            elem_b = elem
            break
    if not elem_b: elem_b = sorted_species[1][0]

    return elem_b, elem_x


def get_bxb_angles_strict(structure, elem_b, elem_x):
    bxb_angles = []
    x_indices = [i for i, s in enumerate(structure.sites) if s.species_string == elem_x]
    b_indices = [i for i, s in enumerate(structure.sites) if s.species_string == elem_b]

    if not x_indices or not b_indices: return []

    for i_x in x_indices:
        site_x = structure.sites[i_x]
        connected_bs = []

        # 找邻居
        for i_b in b_indices:
            dist = site_x.distance(structure.sites[i_b])
            if dist < 3.2:
                connected_bs.append((i_b, dist))

        if len(connected_bs) >= 2:
            connected_bs.sort(key=lambda x: x[1])
            b1_idx, d1 = connected_bs[0]
            b2_idx, d2 = connected_bs[1]

            if b1_idx == b2_idx: continue

            # 计算角度
            pos_x = site_x.coords
            pos_b1 = structure.sites[b1_idx].coords
            pos_b2 = structure.sites[b2_idx].coords

            v1 = pos_b1 - pos_x
            v2 = pos_b2 - pos_x

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0: continue

            cos_theta = np.dot(v1, v2) / (n1 * n2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))

            # 这里的 30 度过滤只是防止计算错误，真正的过滤在下面
            if angle > 30.0:
                bxb_angles.append(angle)

    return bxb_angles


def get_bx_lengths(structure, elem_b, elem_x):
    lengths = []
    b_indices = [i for i, s in enumerate(structure.sites) if s.species_string == elem_b]
    x_indices = [i for i, s in enumerate(structure.sites) if s.species_string == elem_x]

    for i_b in b_indices:
        site_b = structure.sites[i_b]
        for i_x in x_indices:
            dist = site_b.distance(structure.sites[i_x])
            if dist < 3.2:
                lengths.append(dist)
    return lengths


# ================= 主程序 =================
print(f"🔍 正在扫描...")
vasp_files = glob.glob(os.path.join(INPUT_FOLDER, "**/*.vasp"), recursive=True)
if not vasp_files:
    vasp_files = glob.glob(os.path.join(INPUT_FOLDER, "**/CONTCAR"), recursive=True)
print(f"📂 找到 {len(vasp_files)} 个文件")

global_bx_lengths = defaultdict(list)
global_bxb_angles = defaultdict(list)

for fpath in vasp_files:
    try:
        if fpath.endswith('.vasp'):
            struct = Poscar.from_file(fpath).structure
        else:
            struct = Structure.from_file(fpath)

        elem_b, elem_x = identify_B_and_X(struct)
        if not elem_b or not elem_x: continue

        lengths = get_bx_lengths(struct, elem_b, elem_x)
        # 获取所有角度
        all_angles = get_bxb_angles_strict(struct, elem_b, elem_x)

        # <--- 核心修改：在这里进行硬性过滤，只保留 > 160 度的角度
        filtered_angles = [a for a in all_angles if a >= MIN_ANGLE_CUTOFF]

        if lengths:
            global_bx_lengths[f"{elem_x}-{elem_b}"].extend(lengths)

        # 注意：这里使用的是过滤后的角度
        if filtered_angles:
            global_bxb_angles[f"{elem_x}-{elem_b}-{elem_x}"].extend(filtered_angles)

    except Exception as e:
        continue

# ================= 统计与输出 =================
# 筛选 Top 10
# 注意：因为过滤了角度，有些类型的数据量可能会变少，甚至消失
sorted_bonds = sorted(global_bx_lengths.items(), key=lambda x: len(x[1]), reverse=True)[:10]
sorted_angles = sorted(global_bxb_angles.items(), key=lambda x: len(x[1]), reverse=True)[:10]

print(f"\n🏆 Top 10 B-X 键长:")
for k, v in sorted_bonds: print(f"   {k}: {len(v)}")

print(f"\n🏆 Top 10 B-X-B 键角 (仅保留 > 160° 数据):")
for k, v in sorted_angles:
    print(f"   {k}: {len(v)} (Min: {min(v):.1f}, Max: {max(v):.1f})")


# 导出 Excel
def pad_list(lst, target_len):
    return lst + [np.nan] * (target_len - len(lst))


with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
    if sorted_bonds:
        max_len = max(len(v) for _, v in sorted_bonds)
        df_bond = pd.DataFrame({k: pad_list(v, max_len) for k, v in sorted_bonds})
        df_bond.to_excel(writer, sheet_name="Bond_Lengths")

    if sorted_angles:
        max_len = max(len(v) for _, v in sorted_angles)
        df_angle = pd.DataFrame({k: pad_list(v, max_len) for k, v in sorted_angles})
        df_angle.to_excel(writer, sheet_name="Bond_Angles_160-180")

print(f"\n✅ 完成！数据已过滤，仅包含 160-180° 的角度。")