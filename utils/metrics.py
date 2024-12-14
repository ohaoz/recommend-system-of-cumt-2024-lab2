import numpy as np
from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, ndcg_score

class RecommenderMetrics:
    """推荐系统评估指标"""
    
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        计算Precision@K
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            k: 推荐列表长度
            
        Returns:
            Precision@K分数
        """
        if k <= 0:
            return 0.0
            
        # 获取前k个推荐项的索引
        top_k_indices = np.argsort(y_pred)[-k:]
        
        # 计算准确率
        hits = np.sum(y_true[top_k_indices])
        return hits / k
    
    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        计算Recall@K
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            k: 推荐列表长度
            
        Returns:
            Recall@K分数
        """
        if k <= 0:
            return 0.0
            
        # 获取前k个推荐项的索引
        top_k_indices = np.argsort(y_pred)[-k:]
        
        # 计算召回率
        hits = np.sum(y_true[top_k_indices])
        total_relevant = np.sum(y_true)
        
        if total_relevant == 0:
            return 0.0
            
        return hits / total_relevant
    
    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        计算NDCG@K
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            k: 推荐列表长度
            
        Returns:
            NDCG@K分数
        """
        return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)
    
    @staticmethod
    def map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        计算MAP@K (Mean Average Precision)
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            k: 推荐列表长度
            
        Returns:
            MAP@K分数
        """
        if k <= 0:
            return 0.0
            
        # 获取前k个推荐项的索引
        top_k_indices = np.argsort(y_pred)[-k:][::-1]
        
        # 计算累积准确率
        precisions = []
        hits = 0
        
        for i, idx in enumerate(top_k_indices):
            if y_true[idx] == 1:
                hits += 1
                precisions.append(hits / (i + 1))
                
        if not precisions:
            return 0.0
            
        return np.mean(precisions)
    
    @staticmethod
    def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            k: 推荐列表长度
            
        Returns:
            包含所有评估指标的字典
        """
        metrics = {
            f'precision@{k}': RecommenderMetrics.precision_at_k(y_true, y_pred, k),
            f'recall@{k}': RecommenderMetrics.recall_at_k(y_true, y_pred, k),
            f'ndcg@{k}': RecommenderMetrics.ndcg_at_k(y_true, y_pred, k),
            f'map@{k}': RecommenderMetrics.map_at_k(y_true, y_pred, k)
        }
        
        return metrics 