
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
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from pymatgen.core import Structure
import warnings
import json

class MaterialDatasetLoader:
    """材料数据集加载器"""

    def __init__(self, data_dir, id_prop_file):
        self.data_dir = data_dir
        self.id_prop_file = id_prop_file
        self.structures = []
        self.material_ids = []

    def load_dataset(self):
        """加载数据集"""
        try:
            id_prop_df = pd.read_csv(self.id_prop_file, header=None, names=['id', 'band_gap'])
            band_gaps = dict(zip(id_prop_df['id'], id_prop_df['band_gap']))
        except Exception as e:
            print(f'Error loading id_prop.csv: {e}')
            band_gaps = {}
        vasp_files = [f for f in os.listdir(self.data_dir) if f.endswith('.vasp')]
        structures = []
        material_ids = []
        for vasp_file in vasp_files:
            try:
                material_id = vasp_file.replace('.vasp', '')
                structure_path = os.path.join(self.data_dir, vasp_file)
                structure = Structure.from_file(structure_path)
                structure.material_id = material_id
                structure.band_gap = band_gaps.get(material_id, 0.0)
                structures.append(structure)
                material_ids.append(material_id)
            except Exception as e:
                print(f'Error loading {vasp_file}: {e}')
                continue
        self.structures = structures
        self.material_ids = material_ids
        print(f'Successfully loaded {len(structures)} structures')
        return structures

class MaterialFeatureExtractor:
    """材料特征提取器"""

    def __init__(self):
        self.feature_names = None

    def extract_features(self, structure):
        """从单个结构中提取特征"""
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

    def extract_all_features(self, structures):
        """从所有结构中提取特征"""
        features_list = []
        valid_structures = []
        for structure in structures:
            try:
                features = self.extract_features(structure)
                features_list.append(features)
                valid_structures.append(structure)
                if self.feature_names is None:
                    self.feature_names = list(features.keys())
            except Exception as e:
                print(f'Error extracting features from structure: {e}')
                continue
        features_df = pd.DataFrame(features_list, columns=self.feature_names)
        features_df = features_df.fillna(features_df.mean())
        return (features_df, valid_structures)

class MaterialClusterAnalyzer:
    """材料聚类分析器"""

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_centers = None

    def perform_clustering(self, features_df):
        """对材料进行聚类"""
        scaled_features = self.scaler.fit_transform(features_df)
        distance_matrix = squareform(pdist(scaled_features, metric='euclidean'))
        Z = linkage(distance_matrix, method='ward')
        self.cluster_labels = fcluster(Z, t=self.n_clusters, criterion='maxclust')
        self.cluster_centers = []
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = self.cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_center = np.mean(scaled_features[cluster_mask], axis=0)
                self.cluster_centers.append(cluster_center)
            else:
                self.cluster_centers.append(np.zeros(scaled_features.shape[1]))
        return self.cluster_labels

class MaterialWeightCalculator:
    """材料权重计算器"""

    def __init__(self):
        self.weights_data = {}

    def calculate_weights(self, features_df, cluster_labels, target_materials, structures):
        """计算目标材料的权重"""
        material_id_to_index = {}
        for i, structure in enumerate(structures):
            material_id = getattr(structure, 'material_id', f'material_{i}')
            material_id_to_index[material_id] = i
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        cluster_stats = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = scaled_features[cluster_mask]
            if len(cluster_features) > 0:
                cluster_stats[cluster_id] = {'center': np.mean(cluster_features, axis=0), 'size': np.sum(cluster_mask), 'diversity': np.var(cluster_features, axis=0).mean() if len(cluster_features) > 1 else 0}
        weights_info = {}
        for material_id in target_materials:
            if material_id not in material_id_to_index:
                print(f'Warning: Material {material_id} not found in dataset')
                continue
            idx = material_id_to_index[material_id]
            cluster_id = cluster_labels[idx]
            material_features = scaled_features[idx]
            stats = cluster_stats.get(cluster_id, {'center': np.zeros_like(material_features), 'size': 1, 'diversity': 0})
            distance_to_center = np.linalg.norm(material_features - stats['center'])
            representativeness_weight = 1.0 / (1.0 + distance_to_center)
            importance_weight = np.log(stats['size'] + 1) / np.log(len(features_df) + 1)
            diversity_weight = stats['diversity'] / (np.var(scaled_features, axis=0).mean() + 1e-08)
            feature_significance = np.abs(material_features).mean()
            band_gap = features_df.iloc[idx]['band_gap']
            band_gap_weight = (band_gap - features_df['band_gap'].min()) / (features_df['band_gap'].max() - features_df['band_gap'].min() + 1e-08)
            final_weight = 0.35 * representativeness_weight + 0.25 * importance_weight + 0.2 * diversity_weight + 0.1 * feature_significance + 0.1 * band_gap_weight
            weights_info[material_id] = {'cluster_id': int(cluster_id), 'final_weight': float(final_weight), 'representativeness_weight': float(representativeness_weight), 'importance_weight': float(importance_weight), 'diversity_weight': float(diversity_weight), 'feature_significance': float(feature_significance), 'band_gap_weight': float(band_gap_weight), 'distance_to_center': float(distance_to_center), 'cluster_size': int(stats['size']), 'band_gap': float(band_gap), 'volume': float(features_df.iloc[idx]['volume']), 'num_elements': int(features_df.iloc[idx]['num_elements']), 'composition': str(structures[idx].composition)}
        self.weights_data = weights_info
        return weights_info

def save_results(weights_info, output_dir):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    csv_data = []
    for material_id, info in weights_info.items():
        row = {'material_id': material_id, 'cluster_id': info['cluster_id'], 'final_weight': info['final_weight'], 'representativeness_weight': info['representativeness_weight'], 'importance_weight': info['importance_weight'], 'diversity_weight': info['diversity_weight'], 'feature_significance': info['feature_significance'], 'band_gap_weight': info['band_gap_weight'], 'distance_to_center': info['distance_to_center'], 'cluster_size': info['cluster_size'], 'band_gap': info['band_gap'], 'volume': info['volume'], 'num_elements': info['num_elements'], 'composition': info['composition']}
        csv_data.append(row)
    df = pd.DataFrame(csv_data)
    df = df.sort_values('final_weight', ascending=False)
    csv_path = os.path.join(output_dir, 'material_weights_clusters.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    json_data = {}
    for material_id, info in weights_info.items():
        json_data[material_id] = {'cluster_id': info['cluster_id'], 'final_weight': info['final_weight'], 'weight_components': {'representativeness': info['representativeness_weight'], 'importance': info['importance_weight'], 'diversity': info['diversity_weight'], 'feature_significance': info['feature_significance'], 'band_gap': info['band_gap_weight']}, 'cluster_info': {'distance_to_center': info['distance_to_center'], 'cluster_size': info['cluster_size']}, 'material_properties': {'band_gap': info['band_gap'], 'volume': info['volume'], 'num_elements': info['num_elements'], 'composition': info['composition']}}
    json_path = os.path.join(output_dir, 'material_weights_clusters.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    text_path = os.path.join(output_dir, 'summary.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write('Material Weights and Clusters Analysis\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Total materials analyzed: {len(weights_info)}\n')
        f.write(f"Number of clusters: {len(set((info['cluster_id'] for info in weights_info.values())))}\n\n")
        f.write('Materials by Cluster:\n')
        cluster_materials = {}
        for material_id, info in weights_info.items():
            cluster_id = info['cluster_id']
            if cluster_id not in cluster_materials:
                cluster_materials[cluster_id] = []
            cluster_materials[cluster_id].append(material_id)
        for cluster_id in sorted(cluster_materials.keys()):
            f.write(f'\nCluster {cluster_id} ({len(cluster_materials[cluster_id])} materials):\n')
            for material_id in cluster_materials[cluster_id]:
                weight = weights_info[material_id]['final_weight']
                f.write(f'  {material_id}: {weight:.4f}\n')
        f.write(f'\nTop 5 materials by weight:\n')
        sorted_materials = sorted(weights_info.items(), key=lambda x: x[1]['final_weight'], reverse=True)
        for i, (material_id, info) in enumerate(sorted_materials[:5], 1):
            f.write(f"{i}. {material_id}: {info['final_weight']:.4f} (Cluster {info['cluster_id']})\n")
    return df
