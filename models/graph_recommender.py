import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from node2vec import Node2Vec
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from .base_recommender import BaseRecommender

class GraphRecommender(BaseRecommender):
    """基于图的推荐系统"""
    
    def __init__(self, name: str = "GraphBased", embedding_dim: int = 128,
                 walk_length: int = 80, num_walks: int = 10, p: float = 1.0,
                 q: float = 1.0, workers: int = 1):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
            embedding_dim: 嵌入向量维度
            walk_length: 随机游走长度
            num_walks: 每个节点的游走次数
            p: 返回参数
            q: 外出参数
            workers: 并行worker数量
        """
        super().__init__(name)
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        
        # 存储数据
        self.user_tag_matrix = None
        self.item_tag_matrix = None
        self.user_idx_map = None
        self.item_idx_map = None
        self.tag_idx_map = None
        self.user_item_matrix = None
        
        # 图相关
        self.graph = None
        self.embeddings = None
        
    def _build_graph(self):
        """构建异构图网络"""
        self.graph = nx.Graph()
        
        # 添加用户节点
        for user_id in self.user_idx_map.keys():
            self.graph.add_node(f"u_{user_id}", type="user")
            
        # 添加物品节点
        for item_id in self.item_idx_map.keys():
            self.graph.add_node(f"i_{item_id}", type="item")
            
        # 添加标签节点
        for tag_id in self.tag_idx_map.keys():
            self.graph.add_node(f"t_{tag_id}", type="tag")
            
        # 添加用户-标签边
        for user_id, user_idx in self.user_idx_map.items():
            user_tags = self.user_tag_matrix[user_idx]
            for tag_id, tag_idx in self.tag_idx_map.items():
                if user_tags[tag_idx] > 0:
                    self.graph.add_edge(f"u_{user_id}", f"t_{tag_id}",
                                      weight=float(user_tags[tag_idx]))
                    
        # 添加物品-标签边
        for item_id, item_idx in self.item_idx_map.items():
            item_tags = self.item_tag_matrix[item_idx]
            for tag_id, tag_idx in self.tag_idx_map.items():
                if item_tags[tag_idx] > 0:
                    self.graph.add_edge(f"i_{item_id}", f"t_{tag_id}",
                                      weight=float(item_tags[tag_idx]))
                    
        # 添加用户-物品边（如果有评分数据）
        if self.user_item_matrix is not None:
            for user_id, user_idx in self.user_idx_map.items():
                for item_id, item_idx in self.item_idx_map.items():
                    if self.user_item_matrix[user_idx, item_idx] > 0:
                        self.graph.add_edge(f"u_{user_id}", f"i_{item_id}",
                                          weight=float(self.user_item_matrix[user_idx, item_idx]))
    
    def _learn_embeddings(self):
        """学习节点嵌入"""
        # 初始化Node2Vec模型
        node2vec = Node2Vec(
            self.graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers
        )
        
        # 训练模型
        model = node2vec.fit(window=10, min_count=1)
        
        # 保存所有节点的嵌入向量
        self.embeddings = {}
        for node in self.graph.nodes():
            try:
                self.embeddings[node] = model.wv[node]
            except KeyError:
                # 如果节点没有嵌入向量，使用随机向量
                self.embeddings[node] = np.random.randn(self.embedding_dim)
    
    def fit(self, user_tag_matrix: np.ndarray, item_tag_matrix: np.ndarray,
            user_idx_map: Dict, item_idx_map: Dict, tag_idx_map: Dict,
            user_item_matrix: np.ndarray = None) -> 'GraphRecommender':
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
        
        print("构建图网络...")
        self._build_graph()
        
        print("学习节点嵌入...")
        self._learn_embeddings()
        
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
            
        # 获取用户嵌入
        user_emb = self.embeddings.get(f"u_{user_id}")
        if user_emb is None:
            raise ValueError(f"未知用户ID: {user_id}")
            
        # 如果未指定物品列表，则预测所有物品
        if item_ids is None:
            item_ids = list(self.item_idx_map.keys())
            
        # 计算用户和物品的嵌入相似度
        scores = np.zeros(len(item_ids))
        for i, item_id in enumerate(item_ids):
            item_emb = self.embeddings.get(f"i_{item_id}")
            if item_emb is not None:
                scores[i] = np.dot(user_emb, item_emb) / (
                    np.linalg.norm(user_emb) * np.linalg.norm(item_emb))
            else:
                scores[i] = float('-inf')
                
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
        item_ids = list(self.item_idx_map.keys())
        scores = self.predict(user_id, item_ids)
        
        # 如果需要排除已交互物品
        if exclude_seen and self.user_item_matrix is not None:
            user_idx = self.user_idx_map[user_id]
            for i, item_id in enumerate(item_ids):
                item_idx = self.item_idx_map[item_id]
                if self.user_item_matrix[user_idx, item_idx] > 0:
                    scores[i] = float('-inf')
                    
        # 获取分数最高的n_items个物品
        top_indices = np.argsort(scores)[-n_items:][::-1]
        
        # 返回推荐结果
        recommendations = [(item_ids[idx], float(scores[idx])) 
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
            
        # 获取物品嵌入
        item_emb = self.embeddings.get(f"i_{item_id}")
        if item_emb is None:
            raise ValueError(f"未知物品ID: {item_id}")
            
        # 计算与所有物品的相似度
        similarities = []
        for other_id in self.item_idx_map.keys():
            if other_id != item_id:
                other_emb = self.embeddings.get(f"i_{other_id}")
                if other_emb is not None:
                    sim = np.dot(item_emb, other_emb) / (
                        np.linalg.norm(item_emb) * np.linalg.norm(other_emb))
                    similarities.append((other_id, float(sim)))
                    
        # 排序并返回前n_items个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_items]
    
    def get_similar_tags(self, tag_id: str, n_tags: int = 10) -> List[Tuple[str, float]]:
        """
        获取与指定标签最相似的标签
        
        Args:
            tag_id: 标签ID
            n_tags: 返回的标签数量
            
        Returns:
            相似标签列表，每个元素为(标签ID, 相似度分数)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 获取标签嵌入
        tag_emb = self.embeddings.get(f"t_{tag_id}")
        if tag_emb is None:
            raise ValueError(f"未知标签ID: {tag_id}")
            
        # 计算与所有标签的相似度
        similarities = []
        for other_id in self.tag_idx_map.keys():
            if other_id != tag_id:
                other_emb = self.embeddings.get(f"t_{other_id}")
                if other_emb is not None:
                    sim = np.dot(tag_emb, other_emb) / (
                        np.linalg.norm(tag_emb) * np.linalg.norm(other_emb))
                    similarities.append((other_id, float(sim)))
                    
        # 排序并返回前n_tags个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_tags] 