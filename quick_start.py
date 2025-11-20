"""
快速開始指南 - RouteX 旅遊推薦系統
最簡單的使用方式
"""

import torch

try:
    from simple_llm_filter import SimpleLLMFilter
    LLM_FILTER_AVAILABLE = True
except ImportError:
    LLM_FILTER_AVAILABLE = False


def quick_start_without_llm():
    """快速開始 - 不使用LLM過濾"""
    
    print(" 快速開始 - 基本推薦")
    print("=" * 50)
    
    # 1. 初始化推薦器
    from route_aware_recommender import create_route_recommender, OSRMClient
    
    # 創建OSRM客戶端（使用公開OSRM服務器）
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        enable_spatial_index=True,
        enable_async=True
    )
    
    # 設置OSRM客戶端
    recommender.osrm_client = osrm_client
    
    # 2. 獲取推薦
    recommendations = recommender.recommend_on_route(
        user_id=user_id,
        user_history=user_history,
        start_location=start,
        end_location=end,
        activityIntent="旅遊探索",
        top_k=5
    )
    
    # 3. 顯示結果
    print(f"\n 獲得 {len(recommendations)} 個推薦:")
    for i, rec in enumerate(recommendations, 1):
        poi = rec['poi']
        print(f"{i}. {poi['name']} - {rec['score']:.3f}分")
    
    return recommendations


def quick_start_with_llm():
    """快速開始 - 使用LLM過濾（推薦）"""
    
    print("\n\n 快速開始 - LLM智能推薦")
    print("=" * 50)
    
    # 1. 初始化推薦器（啟用LLM）
    from route_aware_recommender import create_route_recommender, OSRMClient
    
    # 創建OSRM客戶端
    osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
    
    recommender = create_route_recommender(
        poi_data_path='datasets/meta-California.json.gz',
        model_checkpoint='models/travel_dlrm.pth',
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        enable_spatial_index=True,
        enable_async=True
    )
    
    # 設置OSRM客戶端
    recommender.osrm_client = osrm_client
    
    #  啟用LLM過濾
    recommender.enable_llm_filter = True
    if LLM_FILTER_AVAILABLE:
        try:
            from simple_llm_filter import SimpleLLMFilter
            recommender.llm_filter = SimpleLLMFilter()
            print(" LLM過濾器已啟用")
        except Exception as e:
            print(f"️ LLM過濾器啟用失敗: {e}")
            recommender.enable_llm_filter = False
    else:
        print("️ LLM過濾器不可用")
        recommender.enable_llm_filter = False
    
    # 2. 獲取推薦（自動使用LLM過濾）
    recommendations = recommender.recommend_on_route(
        user_id='1',
        user_history=[],
        start_location=(37.7749, -122.4194),
        end_location=(37.8199, -122.4783),
        top_k=3
    )
    
    # 3. 顯示結果
    print(f"\n LLM過濾後的推薦:")
    for i, rec in enumerate(recommendations, 1):
        poi = rec['poi']
        llm_status = " LLM審核通過" if rec.get('llm_approved') else ""
        print(f"{i}. {poi['name']} - {rec['score']:.3f}分 {llm_status}")
    
    return recommendations


if __name__ == "__main__":
    print(" RouteX 快速開始指南\n")
    
    # 方式1: 基本推薦（快速）
    quick_start_without_llm()
    
    # 方式2: LLM智能推薦（精準但較慢）
    print("\n⏸️  按 Enter 測試LLM智能推薦...")
    input()
    quick_start_with_llm()
    
    print("\n\n 完成! 查看 demo_complete_usage.py 了解更多功能")
