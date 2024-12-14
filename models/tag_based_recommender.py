import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from .base_recommender import BaseRecommender

class TagBasedRecommender(BaseRecommender):
    """基于标签的推荐系统"""
    
    def __init__(self, name: str = "TagBased"):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
        """
        super().__init__(name)
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'TagBasedRecommender':
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
        
        # 计算用户和物品的标签概率分布
        self.user_tag_dist = self._normalize_matrix(self.user_tag_matrix)
        self.item_tag_dist = self._normalize_matrix(self.item_tag_matrix)
        
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
        
        # 获取用户的标签分布
        user_tag_dist = self.user_tag_dist[user_idx]
        
        # 计算用户和所有物品的相似度
        scores = np.zeros(len(item_indices))
        for i, item_idx in enumerate(item_indices):
            item_tag_dist = self.item_tag_dist[item_idx]
            # 使用余弦相似度计算用户和物品的标签分布相似度
            scores[i] = self._cosine_similarity(user_tag_dist, item_tag_dist)
            
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
        recommendations = [(idx_to_item[idx], scores[idx]) 
                         for idx in top_indices]
        
        return recommendations
    
    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        归一化矩阵，使每行和为1
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            归一化后的矩阵
        """
        row_sums = matrix.sum(axis=1)
        # 避免除以0
        row_sums[row_sums == 0] = 1
        return matrix / row_sums[:, np.newaxis]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            a: 向量a
            b: 向量b
            
        Returns:
            余弦相似度
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(a, b) / (norm_a * norm_b) 