import numpy as np
import pandas as pd
import os

def generate_sample_data(num_users=1000, num_items=2000, sparsity=0.01):
    """
    生成示例数据集
    Args:
        num_users: 用户数量
        num_items: 物品数量
        sparsity: 交互矩阵的稀疏度
    """
    # 创建数据目录
    os.makedirs('data/LastFM', exist_ok=True)
    
    # 生成用户-物品交互数据
    num_interactions = int(num_users * num_items * sparsity)
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids
    })
    interactions_df = interactions_df.drop_duplicates()
    
    # 生成社交关系数据
    num_relations = int(num_users * num_users * 0.01)  # 1%的用户对之间有社交关系
    user_ids_1 = np.random.randint(0, num_users, num_relations)
    user_ids_2 = np.random.randint(0, num_users, num_relations)
    
    social_df = pd.DataFrame({
        'user_id': user_ids_1,
        'friend_id': user_ids_2
    })
    social_df = social_df[social_df['user_id'] != social_df['friend_id']]  # 移除自环
    social_df = social_df.drop_duplicates()
    
    # 保存数据
    interactions_df.to_csv('data/LastFM/interactions.csv', index=False)
    social_df.to_csv('data/LastFM/social.csv', index=False)
    
    print(f"Generated sample dataset:")
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of interactions: {len(interactions_df)}")
    print(f"Number of social relations: {len(social_df)}")

if __name__ == '__main__':
    generate_sample_data() 