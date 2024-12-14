import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.data_loader import MovieLensLoader
from models.item_knn_recommender import ItemKNNRecommender
from utils.metrics import RecommenderMetrics

def main():
    # 加载数据
    print("加载数据...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml-latest-small")
    data_loader = MovieLensLoader(data_path)
    ratings, tags, movies = data_loader.load_data()
    
    # 清理标签数据
    print("清理标签数据...")
    cleaned_tags = data_loader.clean_tags(min_tag_freq=3)
    
    # 构建用户-标签矩阵和物品-标签矩阵
    print("构建特征矩阵...")
    user_tag_matrix, user_idx_map, tag_idx_map = data_loader.get_user_tag_matrix()
    item_tag_matrix, item_idx_map, _ = data_loader.get_item_tag_matrix()
    
    # 构建用户-物品交互矩阵
    print("构建用户-物品交互矩阵...")
    n_users = len(user_idx_map)
    n_items = len(item_idx_map)
    user_item_matrix = np.zeros((n_users, n_items))
    
    for _, row in ratings.iterrows():
        user_idx = user_idx_map.get(row['userId'])
        item_idx = item_idx_map.get(row['movieId'])
        if user_idx is not None and item_idx is not None:
            user_item_matrix[user_idx, item_idx] = row['rating']
    
    # 创建推荐器
    print("初始化推荐器...")
    recommender = ItemKNNRecommender(
        n_neighbors=20,
        min_similarity=0.1,
        normalize=True
    )
    
    # 训练模型
    print("训练模型...")
    recommender.fit(
        user_tag_matrix=user_tag_matrix,
        item_tag_matrix=item_tag_matrix,
        user_idx_map=user_idx_map,
        item_idx_map=item_idx_map,
        tag_idx_map=tag_idx_map,
        user_item_matrix=user_item_matrix
    )
    
    # 为示例用户生成推荐
    print("\n生成推荐结果...")
    test_user_id = list(user_idx_map.keys())[0]  # 选择第一个用户作为测试
    recommendations = recommender.recommend(test_user_id, n_items=10)
    
    # 打印推荐结果
    print(f"\n为用户 {test_user_id} 的推荐结果:")
    print("电影ID\t预测分数\t电影标题")
    print("-" * 50)
    for movie_id, score in recommendations:
        movie_info = movies[movies['movieId'] == movie_id].iloc[0]
        print(f"{movie_id}\t{score:.4f}\t{movie_info['title']}")
    
    # 打印用户的标签使用情况
    print(f"\n用户 {test_user_id} 的标签使用情况:")
    user_idx = user_idx_map[test_user_id]
    user_tags = user_tag_matrix[user_idx]
    tag_counts = [(tag, count) for tag, count in zip(tag_idx_map.keys(), user_tags) if count > 0]
    tag_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("标签\t使用次数")
    print("-" * 30)
    for tag, count in tag_counts[:10]:  # 只显示前10个最常用的标签
        print(f"{tag}\t{count}")
        
    # 分析物品相似度
    if recommendations:  # 如果有推荐结果
        test_item = recommendations[0][0]  # 选择第一个推荐的电影
        similar_items = recommender.get_similar_items(test_item, n_items=5)
        
        print(f"\n与电影 '{movies[movies['movieId'] == test_item].iloc[0]['title']}' 最相似的电影:")
        print("电影ID\t相似度分数\t电影标题")
        print("-" * 50)
        for item_id, score in similar_items:
            movie_info = movies[movies['movieId'] == item_id].iloc[0]
            print(f"{item_id}\t{score:.4f}\t{movie_info['title']}")
            
        # 显示相似电影的标签
        print("\n相似电影的标签分布:")
        for item_id, _ in similar_items:
            movie_info = movies[movies['movieId'] == item_id].iloc[0]
            item_idx = item_idx_map[item_id]
            item_tags = item_tag_matrix[item_idx]
            item_tag_counts = [(tag, count) for tag, count in zip(tag_idx_map.keys(), item_tags) if count > 0]
            item_tag_counts.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n电影 '{movie_info['title']}' 的标签:")
            print("标签\t使用次数")
            print("-" * 30)
            for tag, count in item_tag_counts[:5]:  # 只显示前5个最常用的标签
                print(f"{tag}\t{count}")

if __name__ == "__main__":
    main() 