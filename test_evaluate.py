"""
測試評估程式是否能正常運行
"""

import torch
import numpy as np
from dlrm_model import create_travel_dlrm

print("="*60)
print("測試評估程式 - 模型初始化")
print("="*60)

# 測試模型創建
user_vocab_sizes = {
    'user_id': 10000,
}

poi_vocab_sizes = {
    'poi_id': 1000,
    'primary_category': 50,
    'state': 10,
    'price_level': 5,
    'is_open': 2
}

print("\n1. 創建模型...")
model = create_travel_dlrm(
    user_continuous_dim=10,
    poi_continuous_dim=8,
    path_continuous_dim=4,
    user_vocab_sizes=user_vocab_sizes,
    poi_vocab_sizes=poi_vocab_sizes,
    embedding_dim=64,
    bottom_mlp_dims=[256, 128],
    top_mlp_dims=[512, 256, 128],
    dropout=0.2
)

print(f"✓ 模型創建成功！")
print(f"  參數量: {sum(p.numel() for p in model.parameters()):,}")

# 測試前向傳播
print("\n2. 測試前向傳播...")
batch_size = 4

user_continuous = torch.randn(batch_size, 10)
user_categorical = {
    'user_id': torch.randint(0, 10000, (batch_size,))
}

poi_continuous = torch.randn(batch_size, 8)
poi_categorical = {
    'poi_id': torch.randint(0, 1000, (batch_size,)),
    'primary_category': torch.randint(0, 50, (batch_size,)),
    'state': torch.randint(0, 10, (batch_size,))
}

path_continuous = torch.randn(batch_size, 4)

model.eval()
with torch.no_grad():
    output = model(
        user_continuous,
        user_categorical,
        poi_continuous,
        poi_categorical,
        path_continuous
    )

print(f"✓ 前向傳播成功！")
print(f"  輸出類型: {type(output)}")
if isinstance(output, dict):
    print(f"  輸出鍵: {output.keys()}")
    print(f"  scores 形狀: {output['scores'].shape}")
    print(f"  scores 範例: {output['scores'][:2].squeeze()}")
else:
    print(f"  輸出形狀: {output.shape}")

print("\n" + "="*60)
print("✓ 所有測試通過！")
print("="*60)
