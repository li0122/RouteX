"""
旅行推薦系統使用範例
展示如何使用路徑感知推薦器
"""

import sys
from pathlib import Path

# 添加模組路徑
sys.path.append(str(Path(__file__).parent))

from route_aware_recommender import create_route_recommender, RouteAwareRecommender
from data_processor import load_and_process_data
import json


def example_real_time_recommendation():
    """範例 1: 即時路線推薦"""
    print("="*60)
    print("範例 1: 即時旅行路線推薦")
    print("="*60)
    
    # 1. 初始化推薦器
    print("\n步驟 1: 初始化推薦系統...")
    recommender = create_route_recommender(
        poi_data_path="datasets/meta-California.json.gz",
        device='cpu'
    )
    
    # 2. 設定用戶資料
    user_id = "user_001"
    user_history = [
        {
            'poi_id': 'some_cafe',
            'category': 'cafe',
            'rating': 5.0,
            'visit_date': '2025-01-15'
        },
        {
            'poi_id': 'some_museum',
            'category': 'museum',
            'rating': 4.5,
            'visit_date': '2025-02-20'
        },
        {
            'poi_id': 'some_restaurant',
            'category': 'restaurant',
            'rating': 4.8,
            'visit_date': '2025-03-10'
        }
    ]
    
    # 3. 設定旅行路線 (金門大橋 → 迪士尼樂園)
    print("\n步驟 2: 設定旅行路線...")
    start_location = (37.8199, -122.4783)  # 金門大橋 (舊金山)
    end_location = (33.8121, -117.9190)  # 迪士尼樂園 (安那罕)
    
    print(f"  Start Point: Golden Gate Bridge {start_location}")
    print(f"  End Point: Disneyland Park {end_location}")

    # 4. 獲取推薦
    print("\n步驟 3: 獲取沿途推薦...")
    try:
        recommendations = recommender.recommend_on_route(
            user_id=user_id,
            user_history=user_history,
            start_location=start_location,
            end_location=end_location,
            top_k=5,
            max_detour_ratio=1.25,
            max_extra_duration=900  # 15分鐘
        )
        
        # 5. 顯示推薦結果
        print(f"\n✨ 為您推薦 {len(recommendations)} 個沿途景點:")
        print("-"*60)
        
        for i, rec in enumerate(recommendations, 1):
            poi = rec['poi']
            print(f"\n{i}. {poi['name']}")
            print(f"   評分: {'⭐' * int(poi.get('avg_rating', 0))} {poi.get('avg_rating', 0):.1f}/5.0")
            print(f"   類別: {poi.get('primary_category', 'Unknown')}")
            print(f"   額外時間: {rec['extra_time_minutes']:.0f} 分鐘")
            print(f"   推薦分數: {rec['score']:.3f}")
            print(f"   推薦理由:")
            for reason in rec['reasons']:
                print(f"     • {reason}")
        
        print("\n" + "="*60)
        print("✓ 推薦完成!")
        
    except Exception as e:
        print(f"推薦過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()


def example_location_search():
    """範例 2: 地點搜索"""
    print("\n" + "="*60)
    print("範例 2: 搜索特定地點附近的景點")
    print("="*60)
    
    from data_processor import POIDataProcessor
    
    # 載入數據
    processor = POIDataProcessor("datasets/meta-other.json")
    processor.load_data(max_records=5000)
    processor.preprocess()
    
    # 搜索舊金山市中心附近的景點
    center_location = (37.7749, -122.4194)  # 舊金山市中心
    radius_km = 5.0
    
    print(f"\n搜索位置: 舊金山市中心 {center_location}")
    print(f"搜索半徑: {radius_km} km")
    
    nearby_pois = processor.get_pois_by_location(
        center_location[0], center_location[1], radius_km
    )
    
    print(f"\n找到 {len(nearby_pois)} 個附近景點")
    
    # 顯示前 10 個
    print("\n前 10 個景點:")
    print("-"*60)
    for i, poi in enumerate(nearby_pois[:10], 1):
        print(f"{i}. {poi.get('name', 'Unknown')}")
        print(f"   類別: {poi.get('primary_category', 'Unknown')}")
        print(f"   評分: {poi.get('avg_rating', 0):.1f} ({poi.get('num_reviews', 0)} 評論)")
        
        # 計算距離
        distance = processor._haversine_distance(
            center_location[0], center_location[1],
            poi.get('latitude', 0), poi.get('longitude', 0)
        )
        print(f"   距離: {distance:.2f} km")
        print()


def example_user_preference_analysis():
    """範例 3: 用戶偏好分析"""
    print("="*60)
    print("範例 3: 用戶偏好分析")
    print("="*60)
    
    from route_aware_recommender import UserPreferenceModel
    
    # 創建用戶偏好模型
    preference_model = UserPreferenceModel()
    
    # 模擬用戶歷史
    user_history = [
        {'category': 'cafe', 'rating': 5.0},
        {'category': 'cafe', 'rating': 4.5},
        {'category': 'restaurant', 'rating': 4.8},
        {'category': 'museum', 'rating': 4.0},
        {'category': 'cafe', 'rating': 5.0},
        {'category': 'park', 'rating': 4.5},
    ]
    
    # 建立用戶畫像
    user_profile = preference_model.build_user_profile('user_002', user_history)
    
    print("\n用戶畫像:")
    print("-"*60)
    print(f"用戶ID: {user_profile['user_id']}")
    print(f"平均評分: {user_profile['avg_rating']:.2f}")
    print(f"評分標準差: {user_profile['rating_std']:.2f}")
    print(f"訪問次數: {user_profile['num_visits']}")
    print(f"偏好類別: {', '.join(user_profile['preferred_categories'])}")
    print(f"\n類別分布:")
    for category, count in user_profile['category_distribution'].items():
        percentage = (count / user_profile['num_visits']) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # 獲取用戶特徵向量
    features = preference_model.get_user_features('user_002')
    print(f"\n用戶特徵向量 (前5維): {features[:5]}")


def example_route_planning():
    """範例 4: 路徑規劃與分析"""
    print("\n" + "="*60)
    print("範例 4: 路徑規劃與繞道分析")
    print("="*60)
    
    from route_aware_recommender import OSRMClient
    
    osrm = OSRMClient()
    
    # 定義幾個加州熱門景點
    landmarks = {
        '金門大橋': (37.8199, -122.4783),
        '蒙特雷灣水族館': (36.6180, -121.9016),
        '聖塔莫尼卡碼頭': (34.0094, -118.4973),
        '迪士尼樂園': (33.8121, -117.9190),
        '好萊塢影城': (34.1381, -118.3534)
    }
    
    # 規劃路線: 金門大橋 → 迪士尼樂園
    start = landmarks['金門大橋']
    end = landmarks['迪士尼樂園']
    
    print(f"\n路線規劃: 金門大橋 → 迪士尼樂園")
    print("-"*60)
    
    route = osrm.get_route(start, end)
    if route:
        print(f"直達距離: {route['distance']/1000:.1f} km")
        print(f"預計時間: {route['duration']/60:.0f} 分鐘")
    
    # 測試經過不同景點的繞道
    waypoints = ['蒙特雷灣水族館', '聖塔莫尼卡碼頭']
    
    print(f"\n繞道分析:")
    print("-"*60)
    
    for city_name in waypoints:
        waypoint = landmarks[city_name]
        detour = osrm.calculate_detour(start, waypoint, end)
        
        print(f"\n經過 {city_name}:")
        print(f"  總距離: {detour['via_distance']/1000:.1f} km")
        print(f"  總時間: {detour['via_duration']/60:.0f} 分鐘")
        print(f"  額外距離: {detour['extra_distance']/1000:.1f} km")
        print(f"  額外時間: {detour['extra_duration']/60:.0f} 分鐘")
        print(f"  繞道比例: {detour['detour_ratio']:.2f}x")
        
        # 判斷是否合理
        is_reasonable = detour['detour_ratio'] <= 1.3 and detour['extra_duration'] <= 1800
        status = "✓ 合理繞道" if is_reasonable else "✗ 繞道過遠"
        print(f"  評估: {status}")


def example_batch_recommendation():
    """範例 5: 批量推薦"""
    print("\n" + "="*60)
    print("範例 5: 為多個用戶批量推薦")
    print("="*60)
    
    # 模擬多個用戶
    users = [
        {
            'id': 'user_001',
            'history': [
                {'category': 'cafe', 'rating': 5.0},
                {'category': 'museum', 'rating': 4.5}
            ],
            'route': ((37.8199, -122.4783), (33.8121, -117.9190))  # 金門大橋→迪士尼樂園
        },
        {
            'id': 'user_002',
            'history': [
                {'category': 'restaurant', 'rating': 4.8},
                {'category': 'park', 'rating': 4.0}
            ],
            'route': ((33.8121, -117.9190), (34.1381, -118.3534))  # 迪士尼樂園→好萊塢影城
        }
    ]
    
    print(f"\n為 {len(users)} 個用戶生成推薦...")
    
    # 初始化推薦器
    recommender = create_route_recommender(
        poi_data_path="datasets/meta-California.json.gz",
        model_checkpoint="models/travel_dlrm.pth"
    )
    
    for i, user in enumerate(users, 1):
        print(f"\n{'='*60}")
        print(f"用戶 {i}: {user['id']}")
        print(f"路線: {user['route'][0]} -> {user['route'][1]}")
        print("-"*60)
        
        try:
            recommendations = recommender.recommend_on_route(
                user_id=user['id'],
                user_history=user['history'],
                start_location=user['route'][0],
                end_location=user['route'][1],
                top_k=3
            )
            
            if recommendations:
                for j, rec in enumerate(recommendations, 1):
                    poi = rec['poi']
                    print(f"\n  {j}. {poi['name']}")
                    print(f"     評分: {poi.get('avg_rating', 0):.1f}")
                    print(f"     額外時間: {rec['extra_time_minutes']:.0f} 分鐘")
            else:
                print("  (沒有合適的推薦)")
                
        except Exception as e:
            print(f"  推薦失敗: {e}")


def main():
    """主函數 - 運行所有範例"""
    print("Example.")
    
    examples = [
        ("即時路線推薦", example_real_time_recommendation),
        ("地點搜索", example_location_search),
        ("用戶偏好分析", example_user_preference_analysis),
        ("路徑規劃與分析", example_route_planning),
        # ("批量推薦", example_batch_recommendation),  # 可選
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"正在運行範例 {i}/{len(examples)}: {name}")
            func()
            print(f"\n✓ 範例 {i} 完成!")
        except Exception as e:
            print(f"\n✗ 範例 {i} 失敗: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            print("\n" + "─"*60 + "\n")
    
    print("所有範例演示完成!")


if __name__ == "__main__":
    main()
