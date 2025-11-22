"""
測試評估程式是否能正常運行
"""

import torch
import numpy as np
from dlrm_model import create_travel_dlrm

print("="*60)
print("測試評估程式 - 模型初始化")
print("="*60)

print("\n1. 創建模型...")
# 使用與訓練時一致的配置
user_vocab_sizes = {}  # 訓練時沒有使用用戶類別特徵

poi_vocab_sizes = {
    'category': 50,      # 類別數量
    'state': 10,         # 州數量
    'price_level': 5     # 價格等級
}

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
user_categorical = {}  # 訓練時為空字典

poi_continuous = torch.randn(batch_size, 8)
poi_categorical = {
    'category': torch.randint(0, 50, (batch_size,)),
    'state': torch.randint(0, 10, (batch_size,)),
    'price_level': torch.randint(0, 5, (batch_size,))
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
