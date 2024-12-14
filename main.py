import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import MovieLensLoader
from utils.metrics import RecommenderMetrics

from models.tag_based_recommender import TagBasedRecommender
from models.tensor_recommender import TensorRecommender
from models.lda_recommender import LDARecommender
from models.graph_recommender import GraphRecommender
from models.user_knn_recommender import UserKNNRecommender
from models.item_knn_recommender import ItemKNNRecommender
from models.svd_recommender import SVDRecommender
from models.slope_one_recommender import SlopeOneRecommender

def split_data(ratings_df, test_size=0.2, random_state=42):
    """
    将数据集分割为训练集和测试集
    
    Args:
        ratings_df: 评分数据DataFrame
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        train_df: 训练集
        test_df: 测试集
    """
    return train_test_split(ratings_df, test_size=test_size, random_state=random_state)

def build_matrices(train_df, test_df, data_loader):
    """
    构建训练集和测试集的特征矩阵
    
    Args:
        train_df: 训练集DataFrame
        test_df: 测试集DataFrame
        data_loader: 数据加载器实例
        
    Returns:
        训练集和测试集的特征矩阵和映射字典
    """
    # 清理标签数据
    cleaned_tags = data_loader.clean_tags(min_tag_freq=3)
    
    # 构建用户-标签矩阵和物品-标签矩阵
    user_tag_matrix, user_idx_map, tag_idx_map = data_loader.get_user_tag_matrix()
    item_tag_matrix, item_idx_map, _ = data_loader.get_item_tag_matrix()
    
    # 构建训练集的用户-物品交互矩阵
    n_users = len(user_idx_map)
    n_items = len(item_idx_map)
    train_matrix = np.zeros((n_users, n_items))
    test_matrix = np.zeros((n_users, n_items))
    
    # 填充训练集矩阵
    for _, row in train_df.iterrows():
        user_idx = user_idx_map.get(row['userId'])
        item_idx = item_idx_map.get(row['movieId'])
        if user_idx is not None and item_idx is not None:
            train_matrix[user_idx, item_idx] = row['rating']
            
    # 填充测试集矩阵
    for _, row in test_df.iterrows():
        user_idx = user_idx_map.get(row['userId'])
        item_idx = item_idx_map.get(row['movieId'])
        if user_idx is not None and item_idx is not None:
            test_matrix[user_idx, item_idx] = row['rating']
    
    return (user_tag_matrix, item_tag_matrix, train_matrix, test_matrix,
            user_idx_map, item_idx_map, tag_idx_map)

def evaluate_recommender(recommender, test_matrix, user_idx_map, k=10):
    """
    评估推荐器性能
    
    Args:
        recommender: 推荐器实例
        test_matrix: 测试集矩阵
        user_idx_map: 用户ID到索引的映射
        k: 推荐列表长度
        
    Returns:
        评估指标字典
    """
    precisions = []
    recalls = []
    ndcgs = []
    maps = []
    
    # 对每个用户进行评估
    for user_id in tqdm(user_idx_map.keys(), desc=f"评估 {recommender.name}"):
        user_idx = user_idx_map[user_id]
        actual = test_matrix[user_idx]
        if actual.sum() > 0:  # 只评估在测试集中有评分的用户
            # 获取推荐结果
            recommendations = recommender.recommend(user_id, n_items=k)
            pred = np.zeros_like(actual)
            for item_id, score in recommendations:
                item_idx = recommender.item_idx_map[item_id]
                pred[item_idx] = score
                
            # 计算评估指标
            metrics = RecommenderMetrics.evaluate_all(actual > 0, pred, k)
            precisions.append(metrics[f'precision@{k}'])
            recalls.append(metrics[f'recall@{k}'])
            ndcgs.append(metrics[f'ndcg@{k}'])
            maps.append(metrics[f'map@{k}'])
    
    # 计算平均指标
    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'ndcg@{k}': np.mean(ndcgs),
        f'map@{k}': np.mean(maps)
    }

def plot_results(results, k=10):
    """
    绘制评估结果对比图
    
    Args:
        results: 评估结果字典
        k: 推荐列表长度
    """
    metrics = [f'precision@{k}', f'recall@{k}', f'ndcg@{k}', f'map@{k}']
    algorithms = list(results.keys())
    
    # 创建柱状图
    plt.figure(figsize=(15, 10))
    x = np.arange(len(metrics))
    width = 0.1
    multiplier = 0
    
    for algorithm in algorithms:
        offset = width * multiplier
        rects = plt.bar(x + offset, [results[algorithm][m] for m in metrics],
                       width, label=algorithm)
        multiplier += 1
    
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title(f'推荐算法性能对比 (k={k})')
    plt.xticks(x + width * (len(algorithms) - 1) / 2, metrics)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.close()

def main():
    # 加载数据
    print("加载数据...")
    data_path = "ml-latest-small"
    data_loader = MovieLensLoader(data_path)
    ratings, tags, movies = data_loader.load_data()
    
    # 分割数据集
    print("分割数据集...")
    train_df, test_df = split_data(ratings)
    
    # 构建特征矩阵
    print("构建特征矩阵...")
    matrices = build_matrices(train_df, test_df, data_loader)
    (user_tag_matrix, item_tag_matrix, train_matrix, test_matrix,
     user_idx_map, item_idx_map, tag_idx_map) = matrices
    
    # 定义要评估的推荐器
    recommenders = [
        TagBasedRecommender(),
        TensorRecommender(n_factors=20, n_iterations=50),
        LDARecommender(n_topics=20, max_iter=50),
        GraphRecommender(embedding_dim=128, walk_length=80, num_walks=10),
        UserKNNRecommender(n_neighbors=20, min_similarity=0.1),
        ItemKNNRecommender(n_neighbors=20, min_similarity=0.1),
        SVDRecommender(n_factors=100, normalize=True),
        SlopeOneRecommender(min_common_items=3)
    ]
    
    # 训练和评估每个推荐器
    results = {}
    for recommender in recommenders:
        print(f"\n训练 {recommender.name}...")
        recommender.fit(
            user_tag_matrix=user_tag_matrix,
            item_tag_matrix=item_tag_matrix,
            user_idx_map=user_idx_map,
            item_idx_map=item_idx_map,
            tag_idx_map=tag_idx_map,
            user_item_matrix=train_matrix
        )
        
        print(f"评估 {recommender.name}...")
        metrics = evaluate_recommender(recommender, test_matrix, user_idx_map)
        results[recommender.name] = metrics
        
        # 打印评估结果
        print(f"\n{recommender.name} 评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # 绘制结果对比图
    print("\n绘制结果对比图...")
    plot_results(results)
    
    # 保存结果到CSV
    print("\n保存结果...")
    results_df = pd.DataFrame(results).T
    results_df.to_csv('algorithm_comparison.csv')
    
    # 打印最佳算法
    best_algorithm = max(results.items(), key=lambda x: x[1]['ndcg@10'])
    print(f"\n最佳算法 (基于NDCG@10): {best_algorithm[0]}")
    print("详细指标:")
    for metric, value in best_algorithm[1].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 