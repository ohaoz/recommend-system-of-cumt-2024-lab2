import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from .base_recommender import BaseRecommender

class SlopeOneRecommender(BaseRecommender):
    """基于Slope One的推荐系统"""
    
    def __init__(self, name: str = "SlopeOne", min_common_items: int = 3,
                 min_rating: float = 0.5):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
            min_common_items: 计算物品偏差时的最小共同评分用户数
            min_rating: 最小评分阈值
        """
        super().__init__(name)
        self.min_common_items = min_common_items
        self.min_rating = min_rating
        
        # 存储数据
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        
        # Slope One模型参数
        self.dev_matrix = None  # 物品间的评分偏差矩阵
        self.freq_matrix = None  # 物品间的共同评分用户数矩阵
        
    def _compute_deviation_matrix(self):
        """计算物品间的评分偏差矩阵"""
        n_items = len(self.item_idx_map)
        self.dev_matrix = np.zeros((n_items, n_items))
        self.freq_matrix = np.zeros((n_items, n_items))
        
        # 遍历每对物品
        for i in range(n_items):
            for j in range(i + 1, n_items):
                # 找到同时评分过两个物品的用户
                users_i = np.where(self.user_item_matrix[:, i] > 0)[0]
                users_j = np.where(self.user_item_matrix[:, j] > 0)[0]
                common_users = np.intersect1d(users_i, users_j)
                
                if len(common_users) >= self.min_common_items:
                    # 计算评分偏差
                    ratings_i = self.user_item_matrix[common_users, i]
                    ratings_j = self.user_item_matrix[common_users, j]
                    dev = np.mean(ratings_i - ratings_j)
                    
                    # 存储偏差和频率
                    self.dev_matrix[i, j] = dev
                    self.dev_matrix[j, i] = -dev
                    self.freq_matrix[i, j] = len(common_users)
                    self.freq_matrix[j, i] = len(common_users)
                    
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'SlopeOneRecommender':
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
        
        if self.user_item_matrix is None:
            raise ValueError("Slope One推荐器需要用户-物品交互矩阵")
            
        print("计算物品偏差矩阵...")
        self._compute_deviation_matrix()
        
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
            item_indices = range(len(self.item_idx_map))
        else:
            item_indices = [self.item_idx_map.get(iid) for iid in item_ids]
            if None in item_indices:
                raise ValueError("存在未知的物品ID")
                
        # 获取用户的评分历史
        user_ratings = self.user_item_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return np.zeros(len(item_indices))
            
        # 计算预测评分
        scores = np.zeros(len(item_indices))
        for i, target_idx in enumerate(item_indices):
            if user_ratings[target_idx] > 0:  # 如果用户已经评分过
                scores[i] = user_ratings[target_idx]
            else:
                # 找到用户评分过的物品与目标物品的偏差
                weighted_dev = 0
                total_freq = 0
                
                for rated_idx in rated_items:
                    freq = self.freq_matrix[rated_idx, target_idx]
                    if freq >= self.min_common_items:
                        dev = self.dev_matrix[rated_idx, target_idx]
                        weighted_dev += freq * (user_ratings[rated_idx] + dev)
                        total_freq += freq
                        
                if total_freq > 0:
                    scores[i] = max(weighted_dev / total_freq, self.min_rating)
                else:
                    scores[i] = self.min_rating
                    
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
    
    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        获取与指定物品最相似的物品
        
        Args:
            item_id: 物品ID
            n_items: 返回的物品数量
            
        Returns:
            相似物品列表，每个元素为(物品ID, 相似度分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取物品索引
        item_idx = self.item_idx_map.get(item_id)
        if item_idx is None:
            raise ValueError(f"未知物品ID: {item_id}")
            
        # 计算物品相似度（基于评分偏差和共同评分用户数）
        similarities = np.zeros(len(self.item_idx_map))
        for other_idx in range(len(self.item_idx_map)):
            if other_idx != item_idx:
                freq = self.freq_matrix[item_idx, other_idx]
                if freq >= self.min_common_items:
                    dev = abs(self.dev_matrix[item_idx, other_idx])
                    # 相似度 = 共同评分用户数 / (1 + 评分偏差)
                    similarities[other_idx] = freq / (1 + dev)
                    
        # 获取最相似的n_items个物品
        top_indices = np.argsort(similarities)[-n_items:][::-1]
        
        # 转换回物品ID并返回结果
        idx_to_item = {v: k for k, v in self.item_idx_map.items()}
        similar_items = [(idx_to_item[idx], float(similarities[idx])) 
                        for idx in top_indices]
        
        return similar_items 