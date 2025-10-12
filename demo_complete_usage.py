"""
完整使用範例 - RouteX 旅遊推薦系統 + LLM過濾
展示從初始化到獲得最終推薦的完整流程
"""

import torch
import json
from route_aware_recommender import create_route_recommender, OSRMClient

try:
    from simple_llm_filter import SimpleLLMFilter
    LLM_FILTER_AVAILABLE = True
except ImportError:
    LLM_FILTER_AVAILABLE = False


def example_1_basic_usage():
    """範例1: 基本使用（不使用LLM過濾）"""
    
    print("=" * 60)
    print("📚 範例1: 基本推薦系統使用")
    print("=" * 60)
    
    # 步驟1: 初始化推薦器
    print("\n🔧 步驟1: 初始化推薦器")
    
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_spatial_index=True,
        enable_async=True
    )
    recommender.osrm_client = osrm_client
    recommender.enable_llm_filter = False
    
    print("✅ 推薦器初始化完成")
    
    # 步驟2: 設定用戶資訊和目標位置
    print("\n📍 步驟2: 設定推薦參數")
    user_id = 1
    current_lat = 37.7749  # 舊金山
    current_lon = -122.4194
    destination_lat = 37.8199  # 金門大橋
    destination_lon = -122.4783
    top_k = 5
    
    print(f"   用戶ID: {user_id}")
    print(f"   當前位置: ({current_lat}, {current_lon})")
    print(f"   目的地: ({destination_lat}, {destination_lon})")
    print(f"   需要推薦數: {top_k}")
    
    # 步驟3: 獲取推薦
    print(f"\n🎯 步驟3: 生成推薦")
    recommendations = recommender.recommend_on_route(
        user_id=str(user_id),
        user_history=[],
        start_location=(current_lat, current_lon),
        end_location=(destination_lat, destination_lon),
        top_k=top_k
    )
    
    # 步驟4: 顯示結果
    print(f"\n📊 步驟4: 推薦結果")
    print_recommendations(recommendations)
    
    return recommendations


def example_2_with_llm_filter():
    """範例2: 使用LLM過濾的推薦"""
    
    print("\n\n" + "=" * 60)
    print("🤖 範例2: 使用LLM過濾的推薦系統")
    print("=" * 60)
    
    # 步驟1: 初始化推薦器（啟用LLM）
    print("\n🔧 步驟1: 初始化推薦器（啟用LLM過濾）")
    
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_spatial_index=True,
        enable_async=True
    )
    recommender.osrm_client = osrm_client
    
    # 啟用LLM過濾
    recommender.enable_llm_filter = True
    if LLM_FILTER_AVAILABLE:
        try:
            recommender.llm_filter = SimpleLLMFilter()
            print(f"✅ LLM過濾器已啟用")
        except Exception as e:
            print(f"⚠️ LLM啟用失敗: {e}")
            recommender.enable_llm_filter = False
    
    print("✅ 推薦器初始化完成")
    print(f"   LLM端點: {recommender.llm_filter.base_url if recommender.llm_filter else '未設置'}")
    
    # 步驟2: 設定推薦參數
    print("\n📍 步驟2: 設定推薦參數")
    user_id = 2
    current_lat = 37.7749
    current_lon = -122.4194
    destination_lat = 37.8199
    destination_lon = -122.4783
    top_k = 3
    
    print(f"   用戶ID: {user_id}")
    print(f"   需要推薦數: {top_k}")
    print(f"   💡 系統會用LLM過濾掉不適合旅客的POI")
    
    # 步驟3: 獲取推薦（會自動使用LLM過濾）
    print(f"\n🎯 步驟3: 生成推薦（含LLM審核）")
    recommendations = recommender.recommend_on_route(
        user_id=str(user_id),
        user_history=[],
        start_location=(current_lat, current_lon),
        end_location=(destination_lat, destination_lon),
        top_k=top_k
    )
    
    # 步驟4: 顯示結果
    print(f"\n📊 步驟4: LLM過濾後的推薦結果")
    print_recommendations(recommendations)
    
    return recommendations


def example_3_batch_recommendations():
    """範例3: 批量推薦（多個用戶）"""
    
    print("\n\n" + "=" * 60)
    print("👥 範例3: 批量推薦多個用戶")
    print("=" * 60)
    
    # 初始化推薦器
    print("\n🔧 初始化推薦器")
    
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    recommender.osrm_client = osrm_client
    recommender.enable_llm_filter = True
    if LLM_FILTER_AVAILABLE:
        recommender.llm_filter = SimpleLLMFilter()
    
    # 定義多個用戶的推薦請求
    user_requests = [
        {
            'user_id': 1,
            'current_lat': 37.7749,
            'current_lon': -122.4194,
            'destination_lat': 37.8199,
            'destination_lon': -122.4783,
            'name': '用戶A - 金門大橋遊'
        },
        {
            'user_id': 2,
            'current_lat': 37.7749,
            'current_lon': -122.4194,
            'destination_lat': 37.8086,
            'destination_lon': -122.4098,
            'name': '用戶B - 漁人碼頭遊'
        },
        {
            'user_id': 3,
            'current_lat': 37.7749,
            'current_lon': -122.4194,
            'destination_lat': 37.7694,
            'destination_lon': -122.4862,
            'name': '用戶C - 金門公園遊'
        }
    ]
    
    # 批量處理
    print(f"\n🎯 批量處理 {len(user_requests)} 個推薦請求")
    all_recommendations = {}
    
    for i, request in enumerate(user_requests, 1):
        print(f"\n{'─' * 50}")
        print(f"處理第 {i}/{len(user_requests)} 個請求: {request['name']}")
        
        recommendations = recommender.recommend_on_route(
            user_id=str(request['user_id']),
            user_history=[],
            start_location=(request['current_lat'], request['current_lon']),
            end_location=(request['destination_lat'], request['destination_lon']),
            top_k=3
        )
        
        all_recommendations[request['name']] = recommendations
        print(f"✅ 完成，獲得 {len(recommendations)} 個推薦")
    
    # 顯示所有結果
    print(f"\n\n📊 批量推薦總結")
    print("=" * 60)
    for name, recs in all_recommendations.items():
        print(f"\n{name}:")
        for i, rec in enumerate(recs, 1):
            poi = rec['poi']
            print(f"  {i}. {poi['name']} - {rec['score']:.3f}分")
    
    return all_recommendations


def example_4_custom_user_profile():
    """範例4: 使用自定義用戶偏好"""
    
    print("\n\n" + "=" * 60)
    print("👤 範例4: 自定義用戶偏好推薦")
    print("=" * 60)
    
    # 初始化
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    recommender.osrm_client = osrm_client
    recommender.enable_llm_filter = True
    if LLM_FILTER_AVAILABLE:
        recommender.llm_filter = SimpleLLMFilter()
    
    # 定義用戶偏好
    user_profile = {
        'preferred_categories': ['Restaurant', 'Cafe', 'Museum'],
        'min_rating': 4.0,
        'max_detour_minutes': 20,
        'prefer_popular': True
    }
    
    print("\n👤 用戶偏好設定:")
    print(f"   偏好類別: {', '.join(user_profile['preferred_categories'])}")
    print(f"   最低評分: {user_profile['min_rating']}⭐")
    print(f"   最大繞道時間: {user_profile['max_detour_minutes']}分鐘")
    
    # 獲取推薦
    print("\n🎯 根據偏好生成推薦")
    
    # 注意：user_profile 需要透過 user_history 來建立
    # 這裡我們使用空歷史，實際應用中應該提供真實歷史記錄
    recommendations = recommender.recommend_on_route(
        user_id='100',
        user_history=[],  # 可以新增歷史POI資訊
        start_location=(37.7749, -122.4194),
        end_location=(37.8199, -122.4783),
        top_k=5
    )
    
    # 顯示結果
    print(f"\n📊 個性化推薦結果")
    print_recommendations(recommendations)
    
    return recommendations


def example_5_export_results():
    """範例5: 導出推薦結果到JSON"""
    
    print("\n\n" + "=" * 60)
    print("💾 範例5: 導出推薦結果")
    print("=" * 60)
    
    # 獲取推薦
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    recommender.osrm_client = osrm_client
    recommender.enable_llm_filter = True
    if LLM_FILTER_AVAILABLE:
        recommender.llm_filter = SimpleLLMFilter()
    
    recommendations = recommender.recommend_on_route(
        user_id='1',
        user_history=[],
        start_location=(37.7749, -122.4194),
        end_location=(37.8199, -122.4783),
        top_k=5
    )
    
    # 準備導出數據
    export_data = {
        'user_id': 1,
        'timestamp': '2025-10-12T10:30:00',
        'route': {
            'start': {'lat': 37.7749, 'lon': -122.4194},
            'end': {'lat': 37.8199, 'lon': -122.4783}
        },
        'recommendations': []
    }
    
    # 格式化推薦結果
    for i, rec in enumerate(recommendations, 1):
        poi = rec['poi']
        export_data['recommendations'].append({
            'rank': i,
            'poi_id': poi.get('poi_id', 'unknown'),
            'name': poi['name'],
            'category': poi.get('primary_category', 'Unknown'),
            'rating': poi.get('avg_rating', 0),
            'reviews': poi.get('num_reviews', 0),
            'location': {
                'latitude': poi['latitude'],
                'longitude': poi['longitude']
            },
            'score': rec['score'],
            'extra_time_minutes': rec['extra_time_minutes'],
            'llm_approved': rec.get('llm_approved', False),
            'reasons': rec.get('reasons', [])
        })
    
    # 保存到文件
    output_file = 'recommendations_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 推薦結果已導出到: {output_file}")
    print(f"   包含 {len(recommendations)} 個推薦")
    
    # 顯示預覽
    print(f"\n📄 文件預覽:")
    print(json.dumps(export_data, indent=2, ensure_ascii=False)[:500] + "...")
    
    return export_data


def print_recommendations(recommendations):
    """格式化顯示推薦結果"""
    if not recommendations:
        print("   ⚠️ 沒有推薦結果")
        return
    
    print(f"\n✅ 獲得 {len(recommendations)} 個推薦:")
    print("─" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        poi = rec['poi']
        score = rec['score']
        extra_time = rec['extra_time_minutes']
        llm_approved = rec.get('llm_approved', False)
        
        print(f"\n🎯 推薦 #{i}")
        print(f"   名稱: {poi['name']}")
        print(f"   類別: {poi.get('primary_category', '未分類')}")
        print(f"   評分: {poi.get('avg_rating', 0):.1f}⭐ ({poi.get('num_reviews', 0)} 評論)")
        print(f"   位置: ({poi['latitude']:.4f}, {poi['longitude']:.4f})")
        print(f"   AI評分: {score:.3f}")
        print(f"   額外時間: {extra_time:.1f} 分鐘")
        
        if llm_approved:
            print(f"   ✅ LLM審核: 通過（適合旅客）")
        
        # 顯示推薦理由
        if 'reasons' in rec and rec['reasons']:
            print(f"   推薦理由:")
            for reason in rec['reasons'][:3]:  # 最多顯示3個理由
                print(f"      • {reason}")


def main():
    """主函數 - 執行所有範例"""
    
    print("🎮 RouteX 旅遊推薦系統 - 完整使用範例")
    print("🔗 LLM端點: 140.125.248.15:31008")
    print("🤖 模型: nvidia/llama-3.3-nemotron-super-49b-v1")
    print()
    
    try:
        # 範例1: 基本使用
        example_1_basic_usage()
        
        # 範例2: LLM過濾
        print("\n\n⏸️  按 Enter 繼續下一個範例...")
        input()
        example_2_with_llm_filter()
        
        # 範例3: 批量推薦
        print("\n\n⏸️  按 Enter 繼續下一個範例...")
        input()
        example_3_batch_recommendations()
        
        # 範例4: 自定義偏好
        print("\n\n⏸️  按 Enter 繼續下一個範例...")
        input()
        example_4_custom_user_profile()
        
        # 範例5: 導出結果
        print("\n\n⏸️  按 Enter 繼續下一個範例...")
        input()
        example_5_export_results()
        
        print("\n\n" + "=" * 60)
        print("✅ 所有範例執行完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n\n❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
