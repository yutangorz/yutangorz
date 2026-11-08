
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
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm


def convert_vasp_to_cif_and_id_prop(vasp_dir, output_dir, id_prop_path, recursive=False, overwrite=False):
    """
    将目录中的所有VASP文件转换为CIF格式，并更新id_prop.csv文件

    参数:
    vasp_dir: 包含VASP文件的目录路径
    output_dir: 输出CIF文件的目录路径
    id_prop_path: id_prop.csv文件的路径
    recursive: 是否递归处理子目录
    overwrite: 是否覆盖已存在的CIF文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取id_prop.csv文件
    try:
        id_prop_df = pd.read_csv(id_prop_path, header=None, names=['filename', 'bandgap'])
        print(f"Loaded {len(id_prop_df)} entries from id_prop.csv")
    except Exception as e:
        print(f"Error reading id_prop.csv: {e}")
        return 0

    # 创建新的id_prop.csv数据
    new_id_prop_data = []

    # 获取所有VASP文件
    vasp_extensions = ['.vasp', '.poscar', '.POSCAR', '.contcar', '.CONTCAR']
    vasp_files = []

    if recursive:
        # 递归搜索所有子目录
        for root, dirs, files in os.walk(vasp_dir):
            for file in files:
                # 检查文件扩展名或无扩展名文件
                if any(file.endswith(ext) for ext in vasp_extensions) or '.' not in file:
                    rel_path = os.path.relpath(root, vasp_dir)
                    if rel_path == '.':
                        vasp_files.append((file, root))
                    else:
                        vasp_files.append((os.path.join(rel_path, file), root))
    else:
        # 仅处理当前目录
        for file in os.listdir(vasp_dir):
            if any(file.endswith(ext) for ext in vasp_extensions) or '.' not in file:
                vasp_files.append((file, vasp_dir))

    print(f"Found {len(vasp_files)} VASP files to convert")

    # 转换每个文件
    success_count = 0

    for file_info in tqdm(vasp_files, desc="Converting VASP to CIF"):
        file, root_dir = file_info

        try:
            # 构建完整文件路径
            vasp_path = os.path.join(root_dir, file)

            # 获取基本文件名
            base_name = os.path.splitext(os.path.basename(file))[0]

            # 检查是否需要跳过已存在的文件
            cif_name = f"{base_name}.cif"

            # 处理子目录结构
            if recursive and os.path.dirname(file):
                sub_dir = os.path.join(output_dir, os.path.dirname(file))
                os.makedirs(sub_dir, exist_ok=True)
                cif_path = os.path.join(sub_dir, cif_name)
            else:
                cif_path = os.path.join(output_dir, cif_name)

            # 如果文件已存在且不覆盖，则跳过
            if os.path.exists(cif_path) and not overwrite:
                continue

            # 读取VASP文件
            structure = Structure.from_file(vasp_path)

            # 写入CIF文件
            structure.to(filename=cif_path, fmt="cif")
            success_count += 1

            # 添加到新的id_prop.csv数据
            new_id_prop_data.append(
                [cif_name, id_prop_df.iloc[success_count - 1, 1] if success_count <= len(id_prop_df) else 0])

        except Exception as e:
            print(f"Error converting {file}: {e}")

    # 保存新的id_prop.csv文件
    new_id_prop_df = pd.DataFrame(new_id_prop_data)
    new_id_prop_path = os.path.join(output_dir, "id_prop_cif.csv")
    new_id_prop_df.to_csv(new_id_prop_path, header=False, index=False)
    print(f"Saved new id_prop.csv to: {new_id_prop_path}")

    print(f"Successfully converted {success_count}/{len(vasp_files)} files")
    return success_count


# 在这里指定您的目录路径
if __name__ == "__main__":
    # 设置您的目录路径
    vasp_directory = "vasp_train"  # 替换为您的VASP文件目录路径
    output_directory = "train"  # 替换为您想要保存CIF文件的目录路径
    id_prop_file = "vasp_train/id_prop.csv"  # 替换为您的id_prop.csv文件路径

    # 可选参数
    recursive_processing = False  # 是否递归处理子目录
    overwrite_existing = False  # 是否覆盖已存在的文件

    # 执行转换
    convert_vasp_to_cif_and_id_prop(
        vasp_directory,
        output_directory,
        id_prop_file,
        recursive_processing,
        overwrite_existing
    )