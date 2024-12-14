# MovieLens推荐系统算法比较研究

本项目实现并比较了多种推荐系统算法在MovieLens数据集上的性能表现。项目包含8种不同的推荐算法实现，并提供了完整的评估框架。

## 项目结构

```
recommend/
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── main.py                   # 主程序（算法比较）
├── models/                   # 推荐算法模型
│   ├── base_recommender.py        # 基类
│   ├── tag_based_recommender.py   # 基于标签
│   ├── tensor_recommender.py      # 基于张量分解
│   ├── lda_recommender.py         # 基于LDA主题
│   ├── graph_recommender.py       # 基于图
│   ├── user_knn_recommender.py    # 基于用户KNN
│   ├── item_knn_recommender.py    # 基于物品KNN
│   ├── svd_recommender.py         # 基于SVD
│   └── slope_one_recommender.py   # 基于Slope One
├── examples/                 # 示例代码
│   ├── tag_based_example.py
│   ├── tensor_based_example.py
│   ├── lda_based_example.py
│   ├── graph_based_example.py
│   ├── user_knn_example.py
│   ├── item_knn_example.py
│   ├── svd_example.py
│   └── slope_one_example.py
└── utils/                   # 工具类
    ├── data_loader.py      # 数据加载
    └── metrics.py          # 评估指标
```

## 算法说明

1. **基于标签的推荐 (Tag-based)**
   - 利用用户和物品的标签信息
   - 适合处理冷启动问题
   - 可以提供解释性推荐

2. **基于张量分解的推荐 (Tensor-based)**
   - 将用户-物品-标签关系建模为三维张量
   - 使用CP分解捕获多维特征
   - 可以同时建模多种关系

3. **基于LDA主题模型的推荐 (LDA-based)**
   - 使用LDA发现物品的隐含主题
   - 建模用户的主题偏好
   - 提供可解释的主题特征

4. **基于图的推荐 (Graph-based)**
   - 构建用户-物品-标签异构图
   - 使用node2vec学习节点表示
   - 捕获复杂的网络结构特征

5. **基于用户最近邻的推荐 (User-KNN)**
   - 基于用户相似度进行协同过滤
   - 计算简单，易于实现
   - 推荐结果具有很好的解释性

6. **基于物品最近邻的推荐 (Item-KNN)**
   - 基于物品相似度进行协同过滤
   - 相比User-KNN更稳定
   - 适合物品数量少于用户数量的场景

7. **基于SVD的推荐 (SVD-based)**
   - 使用奇异值分解进行矩阵分解
   - 可以有效处理稀疏数据
   - 计算效率高，扩展性好

8. **基于Slope One的推荐 (Slope One)**
   - 使用加权Slope One算法
   - 考虑评分偏差
   - 简单高效，适合在线推荐

## 评估指标

项目使用以下指标评估算法性能：

- **Precision@K**: 推荐列表中相关物品的比例
- **Recall@K**: 相关物品中被推荐的比例
- **NDCG@K**: 考虑排序位置的归一化折损累积增益
- **MAP@K**: 平均精度均值

## 使用说明

1. 环境配置
```bash
pip install -r requirements.txt
```

2. 数据准备
- 下载MovieLens数据集（ml-latest-small）
- 将数据集放在项目根目录下

3. 运行示例
```bash
# 运行单个算法示例
python examples/tag_based_example.py
python examples/tensor_based_example.py
# ... 其他算法示例

# 运行算法比较
python main.py
```

4. 查看结果
- `algorithm_comparison.png`: 性能对比图
- `algorithm_comparison.csv`: 详细评估结果

## 实验结果

主要实验结果包括：

1. 各算法在不同指标上的表现
2. 算法运行时间对比
3. 推荐结果的多样性分析
4. 冷启动情况下的性能比较

## 扩展性

项目设计考虑了良好的扩展性：

1. 新算法扩展
   - 继承BaseRecommender类
   - 实现必要的接口方法
   - 添加到main.py的评估流程

2. 新指标扩展
   - 在metrics.py中添加新指标
   - 在评估流程中使用新指标

3. 数据集扩展
   - 修改data_loader.py支持新数据集
   - 保持与现有接口兼容

## 注意事项

1. 内存使用
   - 部分算法（如Tensor-based）可能需要较大内存
   - 建议使用小数据集进行测试

2. 运行时间
   - 图算法和张量算法计算较慢
   - 可以调整参数在效果和速度间平衡

3. 参数调优
   - 不同算法有其特定的关键参数
   - 建议根据具体场景调整参数

## 贡献指南

欢迎贡献代码，请遵循以下步骤

1. Fork本项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 参考文献

1. MovieLens数据集: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
2. 各算法相关论文（详见各算法类的文档字符串） 