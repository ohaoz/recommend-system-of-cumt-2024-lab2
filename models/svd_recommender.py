import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .base_recommender import BaseRecommender

class SVDRecommender(BaseRecommender):
    """基于SVD的推荐系统"""
    
    def __init__(self, name: str = "SVD", n_factors: int = 100,
                 normalize: bool = True, min_rating: float = 0.5):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
            n_factors: 隐因子数量
            normalize: 是否对评分进行归一化
            min_rating: 最小评分阈值
        """
        super().__init__(name)
        self.n_factors = n_factors
        self.normalize = normalize
        self.min_rating = min_rating
        
        # 存储数据
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        
        # SVD分解结果
        self.user_features = None
        self.item_features = None
        self.sigma = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        
    def _normalize_ratings(self):
        """对评分进行归一化处理"""
        if not self.normalize:
            return self.user_item_matrix
            
        # 计算全局平均分
        self.global_mean = np.mean(self.user_item_matrix[self.user_item_matrix > 0])
        
        # 计算用户和物品的平均评分
        self.user_means = np.zeros(len(self.user_idx_map))
        self.item_means = np.zeros(len(self.item_idx_map))
        
        for user_idx in range(len(self.user_idx_map)):
            user_ratings = self.user_item_matrix[user_idx]
            if user_ratings.sum() > 0:
                self.user_means[user_idx] = user_ratings[user_ratings > 0].mean()
                
        for item_idx in range(len(self.item_idx_map)):
            item_ratings = self.user_item_matrix[:, item_idx]
            if item_ratings.sum() > 0:
                self.item_means[item_idx] = item_ratings[item_ratings > 0].mean()
                
        # 对评分进行归一化
        normalized_matrix = self.user_item_matrix.copy()
        for user_idx in range(len(self.user_idx_map)):
            for item_idx in range(len(self.item_idx_map)):
                if normalized_matrix[user_idx, item_idx] > 0:
                    normalized_matrix[user_idx, item_idx] -= (
                        self.global_mean +
                        (self.user_means[user_idx] - self.global_mean) +
                        (self.item_means[item_idx] - self.global_mean)
                    )
                    
        return normalized_matrix
        
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'SVDRecommender':
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
            raise ValueError("SVD推荐器需要用户-物品交互矩阵")
            
        print("归一化评分...")
        normalized_matrix = self._normalize_ratings()
        
        print("执行SVD分解...")
        # 使用scipy的svds函数进行SVD分解
        U, s, Vt = svds(normalized_matrix, k=min(self.n_factors,
                                               min(normalized_matrix.shape) - 1))
        
        # 将奇异值转换为对角矩阵
        self.sigma = np.diag(s)
        
        # 保存用户和物品的隐因子矩阵
        self.user_features = U
        self.item_features = Vt.T
        
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
                
        # 计算预测评分
        user_vec = self.user_features[user_idx]
        scores = np.zeros(len(item_indices))
        
        for i, item_idx in enumerate(item_indices):
            item_vec = self.item_features[item_idx]
            score = np.dot(user_vec, np.dot(self.sigma, item_vec))
            
            if self.normalize:
                # 加回偏置项
                score += (self.global_mean +
                         (self.user_means[user_idx] - self.global_mean) +
                         (self.item_means[item_idx] - self.global_mean))
                
            scores[i] = max(score, self.min_rating)
            
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
            
        # 计算物品之间的余弦相似度
        item_vec = self.item_features[item_idx]
        similarities = np.zeros(len(self.item_idx_map))
        
        for other_idx in range(len(self.item_idx_map)):
            if other_idx != item_idx:
                other_vec = self.item_features[other_idx]
                similarity = np.dot(item_vec, other_vec) / (
                    np.linalg.norm(item_vec) * np.linalg.norm(other_vec)
                )
                similarities[other_idx] = similarity
                
        # 获取最相似的n_items个物品
        top_indices = np.argsort(similarities)[-n_items:][::-1]
        
        # 转换回物品ID并返回结果
        idx_to_item = {v: k for k, v in self.item_idx_map.items()}
        similar_items = [(idx_to_item[idx], float(similarities[idx])) 
                        for idx in top_indices]
        
        return similar_items
    
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
            
        # 计算用户之间的余弦相似度
        user_vec = self.user_features[user_idx]
        similarities = np.zeros(len(self.user_idx_map))
        
        for other_idx in range(len(self.user_idx_map)):
            if other_idx != user_idx:
                other_vec = self.user_features[other_idx]
                similarity = np.dot(user_vec, other_vec) / (
                    np.linalg.norm(user_vec) * np.linalg.norm(other_vec)
                )
                similarities[other_idx] = similarity
                
        # 获取最相似的n_users个用户
        top_indices = np.argsort(similarities)[-n_users:][::-1]
        
        # 转换回用户ID并返回结果
        idx_to_user = {v: k for k, v in self.user_idx_map.items()}
        similar_users = [(idx_to_user[idx], float(similarities[idx])) 
                        for idx in top_indices]
        
        return similar_users 