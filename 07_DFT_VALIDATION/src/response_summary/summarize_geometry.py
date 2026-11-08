
# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/response_summary')))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import os
import shutil
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import warnings

class MaterialDatasetLoader:
    """加载材料数据集"""

    def __init__(self, data_dir, id_prop_file):
        """
        初始化数据集加载器

        参数:
        data_dir (str): 包含VASP文件的目录路径
        id_prop_file (str): id_prop.csv文件路径
        """
        self.data_dir = data_dir
        self.id_prop_file = id_prop_file
        self.structures = []
        self.material_ids = []
        self.band_gaps = {}

    def load_dataset(self):
        """加载数据集"""
        print(f'正在从 {self.data_dir} 加载数据集...')
        try:
            id_prop_df = pd.read_csv(self.id_prop_file, header=None, names=['id', 'band_gap'])
            print(f'成功加载 {len(id_prop_df)} 个材料的带隙信息')
            self.band_gaps = dict(zip(id_prop_df['id'], id_prop_df['band_gap']))
        except Exception as e:
            print(f'加载id_prop.csv文件时出错: {e}')
            return None
        vasp_files = [f for f in os.listdir(self.data_dir) if f.endswith('.vasp')]
        print(f'找到 {len(vasp_files)} 个VASP文件')
        structures = []
        material_ids = []
        for i, vasp_file in enumerate(vasp_files):
            if (i + 1) % 100 == 0:
                print(f'已加载 {i + 1}/{len(vasp_files)} 个VASP文件')
            try:
                material_id = vasp_file.replace('.vasp', '')
                structure = Structure.from_file(os.path.join(self.data_dir, vasp_file))
                if material_id in self.band_gaps:
                    structure.band_gap = self.band_gaps[material_id]
                else:
                    structure.band_gap = 0.0
                structure.material_id = material_id
                structures.append(structure)
                material_ids.append(material_id)
            except Exception as e:
                print(f'加载文件 {vasp_file} 时出错: {e}')
                continue
        self.structures = structures
        self.material_ids = material_ids
        print(f'成功加载 {len(structures)} 个材料结构')
        return structures

class MaterialFeatureExtractor:
    """从材料结构中提取特征"""

    def __init__(self):
        self.feature_names = None

    def simple_featurizer(self, structure):
        """简化版特征提取器"""
        features = {}
        comp = structure.composition
        features['num_elements'] = len(comp.elements)
        features['weight'] = comp.weight
        atomic_radii = []
        electronegativities = []
        for e in comp.elements:
            if hasattr(e, 'atomic_radius') and e.atomic_radius:
                atomic_radii.append(e.atomic_radius)
            if hasattr(e, 'X') and e.X:
                electronegativities.append(e.X)
        features['avg_atomic_radius'] = np.mean(atomic_radii) if atomic_radii else 0
        features['avg_electronegativity'] = np.mean(electronegativities) if electronegativities else 0
        features['volume'] = structure.volume
        features['density'] = structure.density
        features['num_sites'] = len(structure)
        lattice = structure.lattice
        features['a'] = lattice.a
        features['b'] = lattice.b
        features['c'] = lattice.c
        features['alpha'] = lattice.alpha
        features['beta'] = lattice.beta
        features['gamma'] = lattice.gamma
        features['volume_per_site'] = lattice.volume / len(structure) if len(structure) > 0 else 0
        features['band_gap'] = getattr(structure, 'band_gap', 0.0)
        return features

    def extract_features(self, structures):
        """从结构列表中提取特征"""
        print('正在提取材料特征...')
        features_list = []
        for i, structure in enumerate(structures):
            if (i + 1) % 100 == 0:
                print(f'已处理 {i + 1}/{len(structures)} 个材料')
            try:
                features = self.simple_featurizer(structure)
                features_list.append(features)
                if self.feature_names is None:
                    self.feature_names = list(features.keys())
            except Exception as e:
                print(f'处理材料 {i} 时出错: {e}')
                continue
        features_df = pd.DataFrame(features_list, columns=self.feature_names)
        features_df = features_df.fillna(features_df.mean())
        print(f'特征提取完成，共提取 {len(features_df)} 个材料的 {len(self.feature_names)} 个特征')
        return features_df

class RepresentativeMaterialSelector:
    """选择代表性材料样本"""

    def __init__(self, n_clusters=10, n_samples_per_cluster=2, max_total_samples=20):
        """
        初始化代表性材料选择器

        参数:
        n_clusters (int): 聚类数量
        n_samples_per_cluster (int): 每个聚类中选择的样本数
        max_total_samples (int): 最大总样本数
        """
        self.n_clusters = n_clusters
        self.n_samples_per_cluster = n_samples_per_cluster
        self.max_total_samples = max_total_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def select_representative_samples(self, features_df, structures):
        """
        选择代表性样本

        参数:
        features_df (pandas.DataFrame): 特征数据框
        structures (list): 对应的结构列表

        返回:
        list: 选中的代表性材料的索引列表
        """
        print('正在选择代表性样本...')
        scaled_features = self.scaler.fit_transform(features_df)
        print('计算距离矩阵...')
        distance_matrix = squareform(pdist(scaled_features, metric='euclidean'))
        print('执行层次聚类...')
        Z = linkage(distance_matrix, method='ward')
        cluster_labels = fcluster(Z, t=self.n_clusters, criterion='maxclust')
        features_df['cluster'] = cluster_labels
        print('计算聚类中心距离...')
        cluster_centers = []
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = features_df['cluster'] == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_center = np.mean(scaled_features[cluster_mask], axis=0)
                cluster_centers.append(cluster_center)
            else:
                cluster_centers.append(np.zeros(scaled_features.shape[1]))
        distances = []
        for i in range(len(scaled_features)):
            cluster_id = cluster_labels[i] - 1
            dist = np.linalg.norm(scaled_features[i] - cluster_centers[cluster_id])
            distances.append(dist)
        features_df['distance_to_center'] = distances
        selected_indices = []
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = features_df['cluster'] == cluster_id
            cluster_indices = features_df[cluster_mask].index
            if len(cluster_indices) == 0:
                continue
            cluster_features = scaled_features[cluster_indices, :]
            cluster_center_distances = features_df.loc[cluster_indices, 'distance_to_center']
            center_sample_idx = cluster_indices[cluster_center_distances.argmin()]
            selected_indices.append(center_sample_idx)
            if self.n_samples_per_cluster > 1 and len(cluster_indices) > 1:
                remaining_indices = [idx for idx in cluster_indices if idx != center_sample_idx]
                for _ in range(min(self.n_samples_per_cluster - 1, len(remaining_indices))):
                    if len(selected_indices) >= self.max_total_samples:
                        break
                    min_distances = []
                    for idx in remaining_indices:
                        dists = cdist(scaled_features[idx].reshape(1, -1), scaled_features[selected_indices], metric='euclidean')
                        min_dist = np.min(dists)
                        min_distances.append(min_dist)
                    farthest_idx = remaining_indices[np.argmax(min_distances)]
                    selected_indices.append(farthest_idx)
                    remaining_indices.remove(farthest_idx)
            if len(selected_indices) >= self.max_total_samples:
                break
        print(f'已选择 {len(selected_indices)} 个代表性材料')
        return selected_indices

class DatasetExporter:
    """导出选中的材料数据集"""

    def __init__(self, output_dir, data_dir):
        """
        初始化数据集导出器

        参数:
        output_dir (str): 输出目录路径
        data_dir (str): 原始数据目录路径
        """
        self.output_dir = output_dir
        self.data_dir = data_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'vasp_files'), exist_ok=True)

    def export_selected_materials(self, selected_indices, structures, features_df, band_gaps):
        """
        导出选中的材料数据集

        参数:
        selected_indices (list): 选中的样本索引
        structures (list): 结构列表
        features_df (pandas.DataFrame): 特征数据框
        band_gaps (dict): 带隙字典
        """
        print(f'正在导出选中的材料到 {self.output_dir}...')
        info_path = os.path.join(self.output_dir, 'selected_materials_info.txt')
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write('Selected Representative Materials\n')
                f.write('=' * 60 + '\n\n')
                for i, idx in enumerate(selected_indices):
                    structure = structures[idx]
                    features = features_df.iloc[idx]
                    material_id = getattr(structure, 'material_id', f'material_{idx}')
                    f.write(f'Material {i + 1}: Index {idx}\n')
                    f.write(f'ID: {material_id}\n')
                    f.write(f'Composition: {structure.composition}\n')
                    f.write(f'Space group: {structure.get_space_group_info()[0]}\n')
                    f.write(f'Density: {structure.density:.4f} g/cm3\n')
                    f.write(f'Volume: {structure.volume:.2f} A3\n')
                    f.write(f"Band gap: {features['band_gap']:.3f} eV\n")
                    f.write(f"Cluster: {features['cluster']}\n")
                    f.write('-' * 60 + '\n')
        except Exception as e:
            print(f'写入材料信息文件时出错: {e}')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write('Selected Representative Materials\n')
                f.write('=' * 60 + '\n\n')
                for i, idx in enumerate(selected_indices):
                    structure = structures[idx]
                    features = features_df.iloc[idx]
                    material_id = getattr(structure, 'material_id', f'material_{idx}')
                    f.write(f'Material {i + 1}: Index {idx}\n')
                    f.write(f'ID: {material_id}\n')
                    f.write(f'Composition: {structure.composition}\n')
                    f.write(f'Space group: {structure.get_space_group_info()[0]}\n')
                    f.write(f'Density: {structure.density:.4f}\n')
                    f.write(f'Volume: {structure.volume:.2f}\n')
                    f.write(f"Band gap: {features['band_gap']:.3f}\n")
                    f.write(f"Cluster: {features['cluster']}\n")
                    f.write('-' * 60 + '\n')
        id_prop_data = []
        for i, idx in enumerate(selected_indices):
            structure = structures[idx]
            material_id = getattr(structure, 'material_id', f'material_{idx}')
            band_gap = band_gaps.get(material_id, 0.0)
            src_file = os.path.join(self.data_dir, f'{material_id}.vasp')
            dst_file = os.path.join(self.output_dir, 'vasp_files', f'{material_id}.vasp')
            if not os.path.exists(src_file):
                try:
                    poscar = Poscar(structure)
                    poscar.write_file(dst_file)
                    print(f'创建VASP文件: {dst_file}')
                except Exception as e:
                    print(f'创建VASP文件 {dst_file} 时出错: {e}')
            else:
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f'复制VASP文件: {src_file} -> {dst_file}')
                except Exception as e:
                    print(f'复制VASP文件 {src_file} 时出错: {e}')
            id_prop_data.append([material_id, band_gap])
        id_prop_df = pd.DataFrame(id_prop_data, columns=['id', 'band_gap'])
        id_prop_path = os.path.join(self.output_dir, 'id_prop.csv')
        try:
            id_prop_df.to_csv(id_prop_path, index=False, header=False, encoding='utf-8')
        except Exception as e:
            print(f'创建id_prop.csv文件时出错: {e}')
            try:
                id_prop_df.to_csv(id_prop_path, index=False, header=False, encoding='gbk')
            except Exception as e:
                print(f'使用GBK编码创建id_prop.csv文件时也出错: {e}')
        print(f'已导出 {len(selected_indices)} 个材料的信息和VASP文件')
        print(f'新的id_prop.csv已创建: {id_prop_path}')
        return id_prop_df
