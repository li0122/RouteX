"""
快速開始指南 - RouteX 旅遊推薦系統
最簡單的使用方式
"""

from route_aware_recommender import RouteAwareRecommender


def quick_start_without_llm():
    """快速開始 - 不使用LLM過濾"""
    
    print("🚀 快速開始 - 基本推薦")
    print("=" * 50)
    
    # 1. 初始化推薦器
    recommender = RouteAwareRecommender(
        model_path='models/travel_dlrm.pth',
        osrm_host='router.project-osrm.org',  # 使用公開OSRM服務器
        osrm_port=80,
        enable_llm_filter=False  # 不使用LLM
    )
    
    # 2. 獲取推薦
    recommendations = recommender.recommend(
        user_id=1,
        current_lat=37.7749,      # 舊金山
        current_lon=-122.4194,
        destination_lat=37.8199,   # 金門大橋
        destination_lon=-122.4783,
        top_k=5
    )
    
    # 3. 顯示結果
    print(f"\n✅ 獲得 {len(recommendations)} 個推薦:")
    for i, rec in enumerate(recommendations, 1):
        poi = rec['poi']
        print(f"{i}. {poi['name']} - {rec['score']:.3f}分")
    
    return recommendations


def quick_start_with_llm():
    """快速開始 - 使用LLM過濾（推薦）"""
    
    print("\n\n🤖 快速開始 - LLM智能推薦")
    print("=" * 50)
    
    # 1. 初始化推薦器（啟用LLM）
    recommender = RouteAwareRecommender(
        model_path='models/travel_dlrm.pth',
        osrm_host='router.project-osrm.org',  # 使用公開OSRM服務器
        osrm_port=80,
        enable_llm_filter=True  # 🔑 啟用LLM過濾
    )
    
    # 2. 獲取推薦（自動使用LLM過濾）
    recommendations = recommender.recommend(
        user_id=1,
        current_lat=37.7749,
        current_lon=-122.4194,
        destination_lat=37.8199,
        destination_lon=-122.4783,
        top_k=3
    )
    
    # 3. 顯示結果
    print(f"\n✅ LLM過濾後的推薦:")
    for i, rec in enumerate(recommendations, 1):
        poi = rec['poi']
        llm_status = "✅ LLM審核通過" if rec.get('llm_approved') else ""
        print(f"{i}. {poi['name']} - {rec['score']:.3f}分 {llm_status}")
    
    return recommendations


if __name__ == "__main__":
    print("🎮 RouteX 快速開始指南\n")
    
    # 方式1: 基本推薦（快速）
    quick_start_without_llm()
    
    # 方式2: LLM智能推薦（精準但較慢）
    print("\n⏸️  按 Enter 測試LLM智能推薦...")
    input()
    quick_start_with_llm()
    
    print("\n\n🎉 完成! 查看 demo_complete_usage.py 了解更多功能")
