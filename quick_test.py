"""
快速測試腳本 - 驗證系統基本功能
"""

import sys
from pathlib import Path

print("="*60)
print("旅行推薦系統 - 快速測試")
print("="*60)

# 測試 1: 導入檢查
print("\n[測試 1] 檢查模組導入...")
try:
    import torch
    print(f"   PyTorch {torch.__version__}")
except ImportError:
    print("   PyTorch 未安裝")
    sys.exit(1)

try:
    import numpy as np
    print(f"   NumPy {np.__version__}")
except ImportError:
    print("   NumPy 未安裝")
    sys.exit(1)

try:
    import requests
    print(f"   Requests")
except ImportError:
    print("   Requests 未安裝")
    sys.exit(1)

# 測試 2: 數據文件檢查
print("\n[測試 2] 檢查數據文件...")

# 檢查多個可能的資料集位置
dataset_locations = [
    ("datasets/meta-other.json", "datasets/review-other.json"),
    ("datasets/meta-California.json.gz", "datasets/review-California.json.gz"),
    ("../datasets/meta-other.json", "../datasets/review-other.json")
]

meta_path = None
review_path = None

for meta, review in dataset_locations:
    meta_p = Path(meta)
    review_p = Path(review)
    if meta_p.exists():
        meta_path = meta_p
        size_mb = meta_path.stat().st_size / (1024 * 1024)
        print(f"   POI 數據: {meta} ({size_mb:.1f} MB)")
        break

if not meta_path:
    print(f"   找不到 POI 數據文件")

for meta, review in dataset_locations:
    review_p = Path(review)
    if review_p.exists():
        review_path = review_p
        size_mb = review_path.stat().st_size / (1024 * 1024)
        print(f"   評論數據: {review} ({size_mb:.1f} MB)")
        break

if not review_path:
    print(f"   找不到評論數據文件")

# 測試 3: DLRM 模型
print("\n[測試 3] 測試 DLRM 模型...")
try:
    from dlrm_model import create_travel_dlrm
    
    model = create_travel_dlrm(
        user_continuous_dim=10,
        poi_continuous_dim=8,
        path_continuous_dim=4,
        user_vocab_sizes={'test': 10},
        poi_vocab_sizes={'category': 100},
        embedding_dim=32
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   DLRM 模型創建成功")
    print(f"   參數量: {num_params:,}")
    
except Exception as e:
    print(f"   DLRM 模型測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 4: 數據處理器
print("\n[測試 4] 測試數據處理器...")
try:
    from data_processor import POIDataProcessor
    
    # 使用之前找到的資料集路徑
    test_path = str(meta_path) if meta_path else "datasets/meta-other.json"
    processor = POIDataProcessor(test_path)
    pois = processor.load_data(max_records=100)
    
    print(f"   載入 {len(pois)} 個 POI")
    
    result = processor.preprocess()
    print(f"   預處理完成")
    print(f"   類別數: {len(processor.category_encoder)}")
    
    if result['pois']:
        sample = result['pois'][0]
        encoded = processor.encode_poi(sample)
        print(f"   POI 編碼成功")
        print(f"     連續特徵維度: {encoded['continuous'].shape}")
        print(f"     類別特徵數: {len(encoded['categorical'])}")
    
except Exception as e:
    print(f"   數據處理器測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 5: OSRM 客戶端
print("\n[測試 5] 測試 OSRM 連接...")
try:
    from route_aware_recommender import OSRMClient
    
    osrm = OSRMClient()
    
    # 測試路徑查詢 (金門大橋 → 迪士尼樂園)
    start = (37.8199, -122.4783)
    end = (33.8121, -117.9190)
    
    print(f"  測試路線: 金門大橋 → 迪士尼樂園")
    route = osrm.get_route(start, end)
    
    if route:
        distance_km = route['distance'] / 1000
        duration_min = route['duration'] / 60
        print(f"   OSRM 連接成功")
        print(f"   距離: {distance_km:.1f} km")
        print(f"   時間: {duration_min:.0f} 分鐘")
        
        # 測試繞道計算
        waypoint = (36.6180, -121.9016)  # 途經蒙特雷灣水族館
        detour = osrm.calculate_detour(start, waypoint, end)
        print(f"   繞道計算成功")
        print(f"     繞道比例: {detour['detour_ratio']:.2f}x")
    else:
        print(f"   OSRM 連接失敗 (可能是網絡問題)")
    
except Exception as e:
    print(f"   OSRM 測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 6: 用戶偏好模型
print("\n[測試 6] 測試用戶偏好模型...")
try:
    from route_aware_recommender import UserPreferenceModel
    
    pref_model = UserPreferenceModel()
    
    user_history = [
        {'category': 'cafe', 'rating': 5.0},
        {'category': 'restaurant', 'rating': 4.5},
        {'category': 'cafe', 'rating': 4.8}
    ]
    
    profile = pref_model.build_user_profile('test_user', user_history)
    features = pref_model.get_user_features('test_user')
    
    print(f"   用戶畫像創建成功")
    print(f"     平均評分: {profile['avg_rating']:.2f}")
    print(f"     偏好類別: {profile['preferred_categories']}")
    print(f"   特徵向量維度: {features.shape}")
    
except Exception as e:
    print(f"   用戶偏好模型測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 總結
print("\n" + "="*60)
print("測試總結")
print("="*60)
print(" 基本功能測試完成!")
print("\n下一步:")
print("  1. 運行完整範例: python example_usage.py")
print("  2. 訓練模型: python train_model.py --max-pois 1000 --epochs 5")
print("  3. 閱讀文檔: cat README.md")
print("="*60)
