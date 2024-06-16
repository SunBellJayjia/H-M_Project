import pandas as pd  
import numpy as np  
import tensorflow as tf  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, concatenate
from tensorflow.keras import EarlyStopping

# 加载数据  
articles_df = pd.read_csv('./base_data/articles.csv')  
customers_df = pd.read_csv('./base_data/customers.csv')  
transactions_df = pd.read_csv('./base_data/transactions_train.csv')

# 数据探索与预处理阶段
# 数据探索与预处理 - articles.csv
# 检查缺失值
print("Articles missing values:\n", articles_df.isnull().sum())
articles_df['detail_desc'].fillna("Unknown", inplace=True)

# 数据探索与预处理 - customers.csv
# 检查缺失值
print("Customers missing values:\n", customers_df.isnull().sum())
# 如果需要填充，可以用一个占位符如"Unknown"
customers_df['FN'].fillna("Unknown", inplace=True)
customers_df['Active'].fillna("Unknown", inplace=True)
customers_df['fashion_news_frequency'].fillna("Unknown", inplace=True)
customers_df['club_member_status'].fillna("Unknown", inplace=True)
customers_df['age'].fillna("Unknown", inplace=True)

# 数据探索与预处理 - transactions_train.csv
# 检查缺失值
print("Transactions missing values:\n", transactions_df.isnull().sum())
# 数据类型转换（如果需要）
# 假设't_dat'是日期字符串，我们需要将其转换为datetime类型
transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
# 可以进一步从日期中提取年、月、日等信息作为新特征
transactions_df['year'] = transactions_df['t_dat'].dt.year
transactions_df['month'] = transactions_df['t_dat'].dt.month
transactions_df['day'] = transactions_df['t_dat'].dt.day

# 展示前几行数据
# 客户数据
print("客户数据：")
print(articles_df.head())
# 打印商品数据前几行
print("商品数据：")
print(customers_df.head())
# 交易数据
print("交易数据：")
print(transactions_df.head())

# 特征工程
# 从articles.csv中提取商品特征
# 假设articles_df中有以下列：'article_id', 'product_type_name', 'product_group_name'
# 提取商品类型和产品组作为分类特征
article_features = articles_df[['article_id', 'product_type_name', 'product_group_name']]

# 为了在后续分析中方便使用，我们可以对分类特征进行编码（例如，使用独热编码）
article_features_encoded = pd.get_dummies(article_features, columns=['product_type_name', 'product_group_name'])
# 从transactions_train.csv中提取与商品相关的交易特征
# 假设transactions_df中有以下列：'transaction_id', 'article_id', 'customer_id', 'price'

# 计算每个商品的总销售额
sales_features = transactions_df.groupby('article_id')['price'].agg(
    {'price': 'sum'}).reset_index()
sales_features.columns = ['article_id', 'total_sales_amount']
# 合并商品特征和交易特征
article_sales_features = pd.merge(article_features_encoded, sales_features, on='article_id')

# （可选）从customers.csv中提取与商品相关的顾客特征，例如哪些顾客群体更喜欢购买某些商品
# 这通常需要更复杂的分析，如关联规则挖掘或协同过滤等。
# 在这里，我们仅简单示范如何根据交易数据将顾客ID添加到商品特征中。

# 假设我们想知道哪些顾客购买了哪些商品
customer_article_mapping = transactions_df[['customer_id', 'article_id']].drop_duplicates()

# 可以进一步分析顾客与商品的关联，但这里我们仅将映射数据保存下来作为示例

# 保存提取的特征以供后续使用
article_sales_features.to_csv('article_sales_features.csv', index=False)
customer_article_mapping.to_csv('customer_article_mapping.csv', index=False)

# 加载处理后的特征数据
article_sales_features = pd.read_csv('article_sales_features.csv')
customer_article_mapping = pd.read_csv('customer_article_mapping.csv')
# 构建用户-物品交互矩阵（这里简化为购买记录）
user_item_interactions = customer_article_mapping.pivot_table(index='customer_id', columns='article_id', aggfunc=len, fill_value=0)
# 将用户和物品ID进行编码
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
user_item_interactions.index = user_encoder.fit_transform(user_item_interactions.index)
user_item_interactions.columns = item_encoder.fit_transform(user_item_interactions.columns)

# 定义超参数搜索空间
hyperParameters = {
    'embedding_dim': [10, 20, 30],  # 嵌入维度
    'learning_rate': [0.001, 0.005, 0.01],  # 学习率
    'batch_size': [32, 64, 128]  # 批处理大小
}

# 划分训练集和测试集
X = user_item_interactions.values
y = np.array([1 if r > 0 else 0 for r in X.flatten()])  # 二分类问题，1表示有交互，0表示无交互
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# 准备训练数据和验证数据（优化：在训练前准备好，而不是在每次迭代中）
user_ids_train = np.repeat(np.arange(X_train.shape[0])[:, np.newaxis], X_train.shape[1], axis=1).flatten()
item_ids_train = np.tile(np.arange(X_train.shape[1]), (X_train.shape[0], 1)).flatten()
labels_train = y_train.flatten()  # 确保标签是一维数组

# # 准备训练数据
# user_ids_train = np.array([np.repeat(i, X_train.shape[1]) for i in range(X_train.shape[0])]).flatten()
# item_ids_train = np.array([list(range(X_train.shape[1])) for _ in range(X_train.shape[0])]).flatten()
# labels_train = y_train

user_ids_val = np.repeat(np.arange(X_val.shape[0])[:, np.newaxis], X_val.shape[1], axis=1).flatten()
item_ids_val = np.tile(np.arange(X_val.shape[1]), (X_val.shape[0], 1)).flatten()
labels_val = y_val.flatten()

# 训练模型并评估的函数
def train_and_evaluate(embedding_dim, learning_rate, batch_size):
    # 定义模型参数
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    # 定义深度学习模型
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_users')(user_embedding)

    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
    item_vec = Flatten(name='flatten_items')(item_embedding)

    dot = tf.keras.layers.dot([user_vec, item_vec], axes=1, normalize=False, name='dot_product')
    model = Model(inputs=[user_input, item_input], outputs=dot)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])
    # 使用tf.data.Dataset创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices(({
                                                            'user_input': user_ids_train,
                                                            'item_input': item_ids_train
                                                        }, labels_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(({
                                                          'user_input': user_ids_val,
                                                          'item_input': item_ids_val
                                                      }, labels_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # 训练模型
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=0)

    # 评估模型
    val_loss, val_acc = model.evaluate(val_dataset)
    return val_loss, val_acc


# 超参数搜索循环
best_val_loss = float('inf')
best_hyperparams = None

for embedding_dim in hyperParameters['embedding_dim']:
    for learning_rate in hyperParameters['learning_rate']:
        for batch_size in hyperParameters['batch_size']:
            val_loss, val_acc = train_and_evaluate(embedding_dim, learning_rate, batch_size)
            print(f'Embedding Dim: {embedding_dim}, Learning Rate: {learning_rate}, Batch Size: {batch_size}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = {'embedding_dim': embedding_dim, 'learning_rate': learning_rate,
                                    'batch_size': batch_size}

print(f'Best herParameters: {best_hyperparams}')
# 使用最佳超参数重新训练模型
train_and_evaluate(best_hyperparams.get('embedding_dim'), best_hyperparams.get('learning_rate'), best_hyperparams.get('batch_size'))



# # 基于深度学习的推荐系统模型
# # 使用PyTorch框架，并假设我们有一个用户-物品交互矩阵作为输入数据
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # 假设用户数量和物品数量
# num_users = 10000
# num_items = 5000
# embedding_dim = 64  # 嵌入维度
#
# # 定义深度学习模型
# class RecommenderModel(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim):
#         super(RecommenderModel, self).__init__()
#         self.user_embeddings = nn.Embedding(num_users, embedding_dim)
#         self.item_embeddings = nn.Embedding(num_items, embedding_dim)
#         self.layers = nn.Sequential(
#             nn.Linear(embedding_dim * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()  # 用于输出预测评分或概率
#         )
#
#     def forward(self, user, item):
#         user_embedding = self.user_embeddings(user)
#         item_embedding = self.item_embeddings(item)
#         combined = torch.cat((user_embedding, item_embedding), 1)
#         output = self.layers(combined)
#         return output
#
#     # 实例化模型
# model = RecommenderModel(num_users, num_items, embedding_dim)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题，如预测评分
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 假设我们有一些训练数据：用户ID和物品ID以及对应的真实评分
# # 这里仅作为示例，实际使用时需要替换为真实数据
# user_ids = torch.randint(0, num_users, (100,))
# item_ids = torch.randint(0, num_items, (100,))
# ratings = torch.rand(100)  # 假设评分在0到1之间
#
# # 训练模型的一个简单循环示例
# num_epochs = 10
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(user_ids, item_ids)
#     loss = criterion(outputs, ratings)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')



# # 使用PyTorch的DistributedDataParallel（DDP）进行分布式训练
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import Dataset, DataLoader
#
# # 假设我们有一个自定义的数据集和数据加载器
# class MyDataset(Dataset):
#     # 实现__len__和__getitem__方法
#     pass
#
# def train(rank, world_size, gpus_per_node):
#     # 初始化进程组
#     dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
#
#     # 设置设备
#     torch.cuda.set_device(rank % gpus_per_node)
#     device = torch.device("cuda:{}".format(rank % gpus_per_node))
#
#     # 创建模型和优化器
#     model = MyModel().to(device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
#     # 使用DistributedDataParallel包装模型
#     model = DistributedDataParallel(model, device_ids=[rank % gpus_per_node])
#
#     # 创建数据加载器
#     dataset = MyDataset()
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#     dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
#
#     # 训练循环
#     for epoch in range(num_epochs):
#         sampler.set_epoch(epoch)  # 设置epoch以确保数据在不同epoch间被打乱
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             # 清理
#     dist.destroy_process_group()
#
# # 假设的模型结构
# class MyModel(torch.nn.Module):
#     # 实现模型结构
#     pass
#
# # 假设的损失函数
# criterion = torch.nn.CrossEntropyLoss()
#
# # 设置超参数
# num_epochs = 10
# world_size = torch.cuda.device_count()  # 假设每个节点使用所有可用的GPU
# gpus_per_node = world_size
#
# # 使用torch.multiprocessing启动多进程训练
# processes = []
# for rank in range(world_size):
#     p = torch.multiprocessing.Process(target=train, args=(rank, world_size, gpus_per_node))
#     p.start()
#     processes.append(p)
#
# for p in processes:
#     p.join()



# from surprise import Dataset, Reader
# from surprise import SVD
# from surprise.model_selection import cross_validate
#
# # 假设transactions_df是从transactions_train.csv加载并处理后的DataFrame
# # 它包含用户ID、物品ID和评分（或购买数量等作为评分），例如：
# # transactions_df = pd.DataFrame({
# #     'customer_id': [1, 1, 2, 2, ...],
# #     'article_id': [1, 2, 1, 3, ...],
# #     'rating': [5, 4, 5, 3, ...]  # 假设购买数量或其他指标作为评分
# # })
#
# # 定义数据读取器并使用DataFrame加载数据
# reader = Reader(rating_scale=(1, 5))  # 假设评分范围是1到5
# data = Dataset.load_from_df(transactions_df[['customer_id', 'article_id', 'price']], reader)
#
# # 使用SVD算法进行协同过滤
# algo = SVD()
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#
# # 训练模型并做出推荐
# trainset = data.build_full_trainset()
# algo.fit(trainset)
#
# # 为特定顾客推荐商品，或者找出与特定商品最相似的其他商品等

# # 数据预处理
# # 对分类特征进行编码
#
# # 加载处理后的特征数据
# article_sales_features = pd.read_csv('article_sales_features.csv')
# customer_article_mapping = pd.read_csv('customer_article_mapping.csv')
#
# # 构建用户-物品交互矩阵（这里简化为购买记录）
# user_item_interactions = customer_article_mapping.pivot_table(index='customer_id', columns='article_id', aggfunc=len,
#                                                               fill_value=0)
#
# # 将用户和物品ID进行编码
# user_encoder = LabelEncoder()
# item_encoder = LabelEncoder()
# user_item_interactions.index = user_encoder.fit_transform(user_item_interactions.index)
# user_item_interactions.columns = item_encoder.fit_transform(user_item_interactions.columns)
#
# # 划分训练集和测试集
# X = user_item_interactions.values
# y = np.array([1 if r > 0 else 0 for r in X.flatten()])  # 二分类问题，1表示有交互，0表示无交互
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 定义模型参数
# num_users = len(user_encoder.classes_)
# num_items = len(item_encoder.classes_)
# embedding_dim = 30
#
# # 定义深度学习模型
# user_input = Input(shape=(1,), dtype='int32', name='user_input')
# item_input = Input(shape=(1,), dtype='int32', name='item_input')
#
# user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
# user_vec = Flatten(name='flatten_users')(user_embedding)
#
# item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
# item_vec = Flatten(name='flatten_items')(item_embedding)
#
# dot = tf.keras.layers.dot([user_vec, item_vec], axes=1, normalize=False, name='dot_product')
# model = Model(inputs=[user_input, item_input], outputs=dot)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])