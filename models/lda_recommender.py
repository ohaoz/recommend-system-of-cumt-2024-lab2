import numpy as np
from typing import Dict, List, Tuple
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import pandas as pd
from .base_recommender import BaseRecommender

class LDARecommender(BaseRecommender):
    """基于LDA主题模型的推荐系统"""
    
    def __init__(self, name: str = "LDABased", n_topics: int = 20, 
                 max_iter: int = 100, learning_method: str = 'batch'):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
            n_topics: 主题数量
            max_iter: 最大迭代次数
            learning_method: 学习方法，'batch'或'online'
        """
        super().__init__(name)
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        
        # 初始化模型
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=42
        )
        
        # 存储数据
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        
        # 存储主题分布
        self.user_topic_dist = None
        self.item_topic_dist = None
        self.tag_topic_dist = None
        
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'LDARecommender':
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
        
        # 对标签矩阵进行LDA分解
        print("对用户-标签矩阵进行LDA分解...")
        self.user_topic_dist = self.lda.fit_transform(self.user_tag_matrix)
        
        # 获取标签-主题分布
        print("计算标签-主题分布...")
        self.tag_topic_dist = self.lda.components_
        
        # 对物品-标签矩阵进行转换
        print("计算物品-主题分布...")
        self.item_topic_dist = self.lda.transform(self.item_tag_matrix)
        
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
        
        # 获取用户的主题分布
        user_topics = self.user_topic_dist[user_idx]
        
        # 计算用户和物品的主题相似度
        scores = np.zeros(len(item_indices))
        for i, item_idx in enumerate(item_indices):
            item_topics = self.item_topic_dist[item_idx]
            # 使用余弦相似度计算主题分布的相似度
            scores[i] = self._cosine_similarity(user_topics, item_topics)
            
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
    
    def get_topic_tags(self, topic_id: int, n_tags: int = 10) -> List[Tuple[str, float]]:
        """
        获取主题下最重要的标签
        
        Args:
            topic_id: 主题ID
            n_tags: 返回的标签数量
            
        Returns:
            标签列表，每个元素为(标签ID, 重要性分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"主题ID必须在0到{self.n_topics-1}之间")
            
        # 获取主题-标签分布
        topic_dist = self.tag_topic_dist[topic_id]
        
        # 获取最重要的标签
        top_indices = np.argsort(topic_dist)[-n_tags:][::-1]
        
        # 转换回标签ID并返回结果
        idx_to_tag = {v: k for k, v in self.tag_idx_map.items()}
        important_tags = [(idx_to_tag[idx], float(topic_dist[idx])) 
                         for idx in top_indices]
        
        return important_tags
    
    def get_user_topics(self, user_id: int, n_topics: int = 5) -> List[Tuple[int, float]]:
        """
        获取用户最感兴趣的主题
        
        Args:
            user_id: 用户ID
            n_topics: 返回的主题数量
            
        Returns:
            主题列表，每个元素为(主题ID, 兴趣度分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取用户索引
        user_idx = self.user_idx_map.get(user_id)
        if user_idx is None:
            raise ValueError(f"未知用户ID: {user_id}")
            
        # 获取用户的主题分布
        user_topics = self.user_topic_dist[user_idx]
        
        # 获取最重要的主题
        top_indices = np.argsort(user_topics)[-n_topics:][::-1]
        
        # 返回主题ID和分数
        return [(idx, float(user_topics[idx])) for idx in top_indices]
    
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