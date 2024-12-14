from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple, Any

class BaseRecommender(ABC):
    """推荐系统基类"""
    
    def __init__(self, name: str):
        """
        初始化推荐器
        
        Args:
            name: 推荐器名称
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> 'BaseRecommender':
        """
        训练模型
        
        Returns:
            self: 训练后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """
        预测用户对物品的兴趣分数
        
        Returns:
            预测分数数组
        """
        pass
    
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
        return []
    
    def evaluate(self, test_data: Any, metrics: List[str], 
                k: int = 10) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            metrics: 评估指标列表
            k: 推荐列表长度
            
        Returns:
            评估结果字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return {}
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        pass 