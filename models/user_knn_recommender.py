import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from .base_recommender import BaseRecommender

class UserKNNRecommender(BaseRecommender):
    """基于用户最近邻的推荐系统"""
    
    def __init__(self, name: str = "UserKNN", n_neighbors: int = 20,
                 min_similarity: float = 0.0, normalize: bool = True):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
            n_neighbors: 近邻数量
            min_similarity: 最小相似度阈值
            normalize: 是否对评分进行归一化
        """
        super().__init__(name)
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.normalize = normalize
        
        # 存储数据
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        
        # 用户特征
        self.user_features = None
        self.knn_model = None
        self.user_means = None
        
    def _build_user_features(self):
        """构建用户特征向量"""
        # 结合评分和标签信息
        if self.user_item_matrix is not None:
            self.user_features = np.hstack([
                self.user_item_matrix,
                self.user_tag_matrix
            ])
        else:
            self.user_features = self.user_tag_matrix
            
        # 计算用户评分均值（用于归一化）
        if self.normalize and self.user_item_matrix is not None:
            self.user_means = np.zeros(len(self.user_idx_map))
            for user_idx in range(len(self.user_idx_map)):
                user_ratings = self.user_item_matrix[user_idx]
                if user_ratings.sum() > 0:
                    self.user_means[user_idx] = user_ratings[user_ratings > 0].mean()
                    
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'UserKNNRecommender':
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
        
        print("构建用户特征...")
        self._build_user_features()
        
        print("训练KNN模型...")
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(self.user_idx_map)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.user_features)
        
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
                
        # 获取用户的近邻
        distances, indices = self.knn_model.kneighbors(
            self.user_features[user_idx].reshape(1, -1)
        )
        
        # 转换距离为相似度
        similarities = 1 - distances[0]
        neighbor_indices = indices[0][1:]  # 排除自身
        neighbor_sims = similarities[1:]
        
        # 过滤掉相似度低于阈值的近邻
        mask = neighbor_sims >= self.min_similarity
        neighbor_indices = neighbor_indices[mask]
        neighbor_sims = neighbor_sims[mask]
        
        if len(neighbor_indices) == 0:
            return np.zeros(len(item_indices))
            
        # 计算预测分数
        scores = np.zeros(len(item_indices))
        if self.user_item_matrix is not None:
            for i, item_idx in enumerate(item_indices):
                # 获取近邻用户对该物品的评分
                neighbor_ratings = self.user_item_matrix[neighbor_indices, item_idx]
                if self.normalize:
                    # 减去近邻用户的平均评分
                    neighbor_ratings = neighbor_ratings - self.user_means[neighbor_indices]
                    
                # 计算加权平均分数
                mask = neighbor_ratings > 0
                if mask.sum() > 0:
                    scores[i] = np.average(
                        neighbor_ratings[mask],
                        weights=neighbor_sims[mask]
                    )
                    if self.normalize:
                        # 加上目标用户的平均评分
                        scores[i] += self.user_means[user_idx]
                        
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
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> List[Tuple[int, float]]:
        """
        获取与指定用户最相似的用户
        
        Args:
            user_id: 用户ID
            n_users: 返回的用户数量
            
        Returns:
            相似用户列表，每个元素为(用户ID, 相似度分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取用户索引
        user_idx = self.user_idx_map.get(user_id)
        if user_idx is None:
            raise ValueError(f"未知用户ID: {user_id}")
            
        # 获取用户的近邻
        distances, indices = self.knn_model.kneighbors(
            self.user_features[user_idx].reshape(1, -1)
        )
        
        # 转换距离为相似度
        similarities = 1 - distances[0]
        neighbor_indices = indices[0][1:n_users+1]  # 排除自身
        neighbor_sims = similarities[1:n_users+1]
        
        # 转换回用户ID
        idx_to_user = {v: k for k, v in self.user_idx_map.items()}
        similar_users = [(idx_to_user[idx], float(sim)) 
                        for idx, sim in zip(neighbor_indices, neighbor_sims)]
        
        return similar_users 