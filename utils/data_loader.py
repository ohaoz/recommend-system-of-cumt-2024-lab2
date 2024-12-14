import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os

class MovieLensLoader:
    """MovieLens数据集加载器"""
    
    def __init__(self, data_path: str = "../ml-latest-small"):
        """
        初始化数据加载器
        
        Args:
            data_path: MovieLens数据集路径
        """
        self.data_path = data_path
        self.ratings = None
        self.tags = None
        self.movies = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载所有必要的数据文件
        
        Returns:
            ratings_df: 评分数据
            tags_df: 标签数据
            movies_df: 电影数据
        """
        # 加载评分数据
        self.ratings = pd.read_csv(os.path.join(self.data_path, "ratings.csv"))
        
        # 加载标签数据
        self.tags = pd.read_csv(os.path.join(self.data_path, "tags.csv"))
        
        # 加载电影数据
        self.movies = pd.read_csv(os.path.join(self.data_path, "movies.csv"))
        
        return self.ratings, self.tags, self.movies
    
    def clean_tags(self, min_tag_freq: int = 3) -> pd.DataFrame:
        """
        清理标签数据
        
        Args:
            min_tag_freq: 最小标签频率阈值
            
        Returns:
            清理后的标签数据
        """
        if self.tags is None:
            self.load_data()
            
        # 转换为小写
        self.tags['tag'] = self.tags['tag'].str.lower()
        
        # 移除特殊字符
        self.tags['tag'] = self.tags['tag'].str.replace('[^\w\s]', '')
        
        # 移除多余的空格
        self.tags['tag'] = self.tags['tag'].str.strip()
        
        # 移除空标签
        self.tags = self.tags[self.tags['tag'].str.len() > 0]
        
        # 计算标签频率
        tag_counts = self.tags['tag'].value_counts()
        
        # 只保留出现频率超过阈值的标签
        valid_tags = tag_counts[tag_counts >= min_tag_freq].index
        self.tags = self.tags[self.tags['tag'].isin(valid_tags)]
        
        return self.tags
    
    def get_user_tag_matrix(self) -> Tuple[np.ndarray, Dict, Dict]:
        """
        构建用户-标签矩阵
        
        Returns:
            user_tag_matrix: 用户-标签矩阵
            user_idx_map: 用户ID到索引的映射
            tag_idx_map: 标签到索引的映射
        """
        if self.tags is None:
            self.load_data()
            
        # 创建用户和标签的索引映射
        unique_users = sorted(self.tags['userId'].unique())
        unique_tags = sorted(self.tags['tag'].unique())
        
        user_idx_map = {uid: idx for idx, uid in enumerate(unique_users)}
        tag_idx_map = {tag: idx for idx, tag in enumerate(unique_tags)}
        
        # 构建用户-标签矩阵
        user_tag_matrix = np.zeros((len(unique_users), len(unique_tags)))
        
        for _, row in self.tags.iterrows():
            user_idx = user_idx_map[row['userId']]
            tag_idx = tag_idx_map[row['tag']]
            user_tag_matrix[user_idx, tag_idx] += 1
            
        return user_tag_matrix, user_idx_map, tag_idx_map
    
    def get_item_tag_matrix(self) -> Tuple[np.ndarray, Dict, Dict]:
        """
        构建物品-标签矩阵
        
        Returns:
            item_tag_matrix: 物品-标签矩阵
            item_idx_map: 物品ID到索引的映射
            tag_idx_map: 标签到索引的映射
        """
        if self.tags is None:
            self.load_data()
            
        # 创建物品和标签的索引映射
        unique_items = sorted(self.tags['movieId'].unique())
        unique_tags = sorted(self.tags['tag'].unique())
        
        item_idx_map = {iid: idx for idx, iid in enumerate(unique_items)}
        tag_idx_map = {tag: idx for idx, tag in enumerate(unique_tags)}
        
        # 构建物品-标签矩阵
        item_tag_matrix = np.zeros((len(unique_items), len(unique_tags)))
        
        for _, row in self.tags.iterrows():
            item_idx = item_idx_map[row['movieId']]
            tag_idx = tag_idx_map[row['tag']]
            item_tag_matrix[item_idx, tag_idx] += 1
            
        return item_tag_matrix, item_idx_map, tag_idx_map 