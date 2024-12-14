import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
import tensorly as tl
from .base_recommender import BaseRecommender

class TensorRecommender(BaseRecommender):
    """基于张量分解的推荐系统"""
    
    def __init__(self, name: str = "TensorBased", n_factors: int = 20, n_iterations: int = 100):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
            n_factors: 隐因子数量
            n_iterations: 迭代次数
        """
        super().__init__(name)
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        self.tensor = None
        self.weights = None
        self.factors = None
        
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'TensorRecommender':
        """
        训练模型
        
        Args:
            user_tag_matrix: 用户-标签矩阵
            item_tag_matrix: 物品-标签矩阵
            user_idx_map: 用户ID到索引的映射
            item_idx_map: 物品ID到索引的映射
            tag_idx_map: 标签到索引的映射
            user_item_matrix: 用户-物品交互矩阵（可选）
            
        Returns:
            self: 训练后的模型实例
        """
        self.user_tag_matrix = user_tag_matrix
        self.item_tag_matrix = item_tag_matrix
        self.user_idx_map = user_idx_map
        self.item_idx_map = item_idx_map
        self.tag_idx_map = tag_idx_map
        self.user_item_matrix = user_item_matrix
        
        # 构建用户-物品-标签三维张量
        n_users = len(user_idx_map)
        n_items = len(item_idx_map)
        n_tags = len(tag_idx_map)
        
        # 初始化张量
        self.tensor = np.zeros((n_users, n_items, n_tags))
        
        # 填充张量
        for user_id, user_idx in user_idx_map.items():
            user_tags = user_tag_matrix[user_idx]
            for item_id, item_idx in item_idx_map.items():
                item_tags = item_tag_matrix[item_idx]
                # 用户和物品的共同标签权重
                for tag_id, tag_idx in tag_idx_map.items():
                    if user_tags[tag_idx] > 0 and item_tags[tag_idx] > 0:
                        self.tensor[user_idx, item_idx, tag_idx] = 1
        
        # 执行CP分解
        tl.set_backend('numpy')
        self.weights, self.factors = parafac(self.tensor, 
                                           rank=self.n_factors,
                                           n_iter_max=self.n_iterations,
                                           return_errors=False)
        
        self.is_fitted = True
        return self
        
    def predict(self, user_id: int, item_ids: List[int] = None) -> np.ndarray:
        """
        预测用户对物品的兴趣分数
        
        Args:
            user_id: 用户ID
            item_ids: 待预测物品ID列表，如果为None则预测所有物品
            
        Returns:
            预测分数数组
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取用户索引
        user_idx = self.user_idx_map.get(user_id)
        if user_idx is None:
            raise ValueError(f"未知用户ID: {user_id}")
            
        # 如果未指定物品列表，则预测所有物品
        if item_ids is None:
            item_indices = range(self.item_tag_matrix.shape[0])
        else:
            item_indices = [self.item_idx_map.get(iid) for iid in item_ids]
            if None in item_indices:
                raise ValueError("存在未知的物品ID")
        
        # 重构张量
        reconstructed_tensor = cp_to_tensor((self.weights, self.factors))
        
        # 获取用户的预测分数
        user_predictions = reconstructed_tensor[user_idx]
        
        # 计算每个物品的总分（所有标签的得分之和）
        scores = np.sum(user_predictions[item_indices], axis=1)
        
        return scores
    
    def recommend(self, user_id: int, n_items: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        为用户推荐物品
        
        Args:
            user_id: 用户ID
            n_items: 推荐物品数量
            exclude_seen: 是否排除用户已交互的物品
            
        Returns:
            推荐物品列表，每个元素为(物品ID, 预测分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 预测所有物品的分数
        scores = self.predict(user_id)
        
        # 获取用户索引
        user_idx = self.user_idx_map[user_id]
        
        # 如果需要排除已交互物品
        if exclude_seen and self.user_item_matrix is not None:
            seen_mask = self.user_item_matrix[user_idx] > 0
            scores[seen_mask] = -np.inf
            
        # 获取分数最高的n_items个物品
        top_indices = np.argsort(scores)[-n_items:][::-1]
        
        # 转换回物品ID并返回推荐结果
        idx_to_item = {v: k for k, v in self.item_idx_map.items()}
        recommendations = [(idx_to_item[idx], float(scores[idx])) 
                         for idx in top_indices]
        
        return recommendations
    
    def get_tag_factors(self) -> np.ndarray:
        """
        获取标签的隐因子表示
        
        Returns:
            标签的隐因子矩阵
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.factors[2]
    
    def get_similar_tags(self, tag_id: str, n_tags: int = 10) -> List[Tuple[str, float]]:
        """
        获取与指定标签最相似的标签
        
        Args:
            tag_id: 标签ID
            n_tags: 返回的相似标签数量
            
        Returns:
            相似标签列表，每个元素为(标签ID, 相似度分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取标签的隐因子表示
        tag_factors = self.get_tag_factors()
        
        # 获取目标标签的索引
        tag_idx = self.tag_idx_map.get(tag_id)
        if tag_idx is None:
            raise ValueError(f"未知标签ID: {tag_id}")
            
        # 计算目标标签与所有标签的相似度
        target_factor = tag_factors[tag_idx]
        similarities = np.dot(tag_factors, target_factor)
        
        # 获取最相似的标签
        top_indices = np.argsort(similarities)[-n_tags-1:][::-1]
        
        # 移除自身
        top_indices = top_indices[top_indices != tag_idx][:n_tags]
        
        # 转换回标签ID并返回结果
        idx_to_tag = {v: k for k, v in self.tag_idx_map.items()}
        similar_tags = [(idx_to_tag[idx], float(similarities[idx])) 
                       for idx in top_indices]
        
        return similar_tags 