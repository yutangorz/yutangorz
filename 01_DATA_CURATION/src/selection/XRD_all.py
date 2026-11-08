
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
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import warnings
import os
from scipy.stats import norm
from tqdm import tqdm
import traceback
from io import StringIO


def parse_cif_string(cif_string):
    """从CIF字符串解析结构"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 使用StringIO将字符串转换为文件对象
            cif_io = StringIO(cif_string)
            parser = CifParser(cif_io)
            structures = parser.parse_structures(primitive=True)
            return structures[0]  # 返回第一个结构
    except Exception as e:
        return None


def calculate_xrd(structure, wavelength='CuKa', min_angle=5, max_angle=90):
    """计算XRD图谱并返回详细数据"""
    if structure is None:
        return None

    try:
        calculator = XRDCalculator(wavelength=wavelength)
        pattern = calculator.get_pattern(structure, two_theta_range=(min_angle, max_angle))
        return pattern
    except Exception as e:
        return None


def generate_continuous_pattern(pattern, min_angle=5, max_angle=90, resolution=0.2, peak_width=0.25):
    """
    生成连续的XRD图谱
    Args:
        pattern: XRDPattern对象
        min_angle: 最小2θ角度
        max_angle: 最大2θ角度
        resolution: 角度分辨率（度）
        peak_width: 峰宽参数（标准差），控制峰宽度
    Returns:
        连续的角度数组，连续的强度数组
    """
    # 创建角度轴
    angles = np.arange(min_angle, max_angle + resolution, resolution)

    # 如果没有峰数据，返回原始角度和零强度
    if pattern is None or len(pattern.x) == 0:
        return angles, np.zeros_like(angles)

    # 创建强度数组
    intensities = np.zeros_like(angles)

    # 对每个峰应用高斯分布函数
    for angle, intensity in zip(pattern.x, pattern.y):
        # 在2θ轴上添加高斯峰
        gaussian = norm(loc=angle, scale=peak_width)
        contrib = intensity * gaussian.pdf(angles)
        intensities += contrib

    # 归一化强度
    max_intensity = np.max(intensities)
    if max_intensity > 0:
        intensities = 100 * intensities / max_intensity
    else:
        intensities = np.zeros_like(intensities)  # 确保不包含NaN值

    # 应用阈值处理：将所有小于0.01的值设为0
    intensities[intensities < 0.49] = 0

    return angles, intensities


def process_all_cifs(csv_path, radiation='CuKa', min_angle=5, max_angle=90, resolution=0.2):
    """
    处理CSV文件中的所有CIF路径
    Args:
        csv_path: 输入的CSV文件路径
        radiation: X射线类型
        min_angle: 最小角度
        max_angle: 最大角度
        resolution: 角度分辨率
    """
    # 1. 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件: {csv_path}")
        print(f"数据集包含 {len(df)} 行")
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")
        return None

    # 检查是否有cif列
    if 'cif' not in df.columns:
        print("错误: CSV文件中没有 'cif' 列")
        return None

    # 2. 创建固定的角度数组
    angles = np.arange(min_angle, max_angle + resolution, resolution)
    angle_columns = [f"{angle:.1f}°" for angle in angles]
    print(f"生成角度列: 从 {min_angle}° 到 {max_angle}°, 分辨率 {resolution}°, 共 {len(angles)} 列")

    # 3. 创建输出目录
    output_dir = os.path.join(os.path.dirname(csv_path), "XRD_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")

    # 4. 创建存储所有XRD数据的矩阵（初始化为NaN）
    xrd_matrix = np.full((len(df), len(angles)), np.nan)

    # 5. 初始化状态列
    status_column = ['未处理'] * len(df)

    # 6. 处理每一行
    print(f"\n开始处理 {len(df)} 个CIF字符串...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        cif_content = row['cif']

        try:
            # 检查CIF内容是否为空
            if pd.isna(cif_content) or not cif_content:
                status_column[idx] = 'CIF内容为空'
                continue

            # 解析CIF字符串
            structure = parse_cif_string(str(cif_content))  # 确保转换为字符串
            if structure is None:
                status_column[idx] = '解析失败'
                continue

            # 计算XRD
            pattern = calculate_xrd(structure, wavelength=radiation,
                                    min_angle=min_angle, max_angle=max_angle)

            if pattern is None:
                status_column[idx] = 'XRD计算失败'
                continue

            # 生成连续谱
            _, intensities = generate_continuous_pattern(pattern, min_angle, max_angle, resolution)

            # 确保强度数组与角度列长度匹配
            if len(intensities) != len(angles):
                status_column[idx] = '长度不匹配'
                continue

            # 检查强度值是否全为零
            if np.max(intensities) == 0:
                status_column[idx] = '强度计算异常'
                continue

            # 将结果保存到矩阵
            xrd_matrix[idx] = intensities

            # 标记为成功
            status_column[idx] = '成功'

        except Exception as e:
            status_column[idx] = f'错误: {str(e)[:50]}'

    # 7. 将XRD矩阵转换为DataFrame
    xrd_df = pd.DataFrame(xrd_matrix, columns=angle_columns)

    # 8. 将状态列添加到原始DataFrame
    df['xrd_status'] = status_column

    # 9. 准备输出数据
    output_columns = [col for col in df.columns if col != 'cif']
    result_df = pd.concat([df[output_columns], xrd_df], axis=1)

    # 10. 保存结果
    output_csv = os.path.join(output_dir, os.path.basename(csv_path).replace('.csv', '_with_XRD.csv'))
    result_df.to_csv(output_csv, index=False)

    # 11. 统计成功和失败的数量
    success_count = np.sum([status == '成功' for status in status_column])
    error_count = len(df) - success_count

    print(f"\n处理完成! 结果已保存到: {output_csv}")
    print(f"成功处理: {success_count}个, 失败: {error_count}个")

    # 返回前几行作为预览
    if success_count > 0:
        print("\n成功样本的前5个角度点强度预览：")
        preview_df = result_df[result_df['xrd_status'] == '成功'].head(3)

        # 检查预览列是否存在
        preview_cols = [col for col in ['entry_id', 'name'] + angle_columns[:5] if col in preview_df.columns]
        if preview_cols:
            preview_df = preview_df[preview_cols]

            # 格式化输出
            for i, row in preview_df.iterrows():
                entry_id = row.get('entry_id', 'N/A')
                name = row.get('name', 'N/A')
                print(f"\nentry_id: {entry_id}, name: {name}")
                for col in angle_columns[:5]:
                    if col in preview_df.columns:
                        print(f"  {col}: {row[col]:.4f}")

    return result_df


if __name__ == "__main__":
    print("===== XRD批处理程序 =====")
    print("功能: 批量处理CSV文件中的CIF字符串，添加XRD数据列")
    print("注意: 最终结果将不包含cif列")

    # 配置参数
    csv_file = str(_repo_path('external_inputs/original_workspace/OQMD/Dateset/DPVK-OQMD-cif_processed_Conventional_St.csv'))
    radiation_type = 'CuKa'  # 可以选择: 'CuKa', 'MoKa', 'AgKa'
    min_angle = 5  # 起始角度（度）
    max_angle = 90  # 结束角度（度）
    resolution = 0.1  # 角度分辨率（度）

    print(f"\n输入文件: {csv_file}")
    print(f"射线源: {radiation_type}")
    print(f"角度范围: {min_angle}° to {max_angle}°, 分辨率: {resolution}°")
    print("=" * 50)

    # 运行批处理
    result_df = process_all_cifs(csv_file, radiation_type, min_angle, max_angle, resolution)

    print("\n批处理完成!")