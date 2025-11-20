"""
簡單測試腳本 - 驗證系統運行
"""

def test_basic_import():
    """測試基本導入"""
    print(" 測試1: 基本導入")
    try:
        from route_aware_recommender import create_route_recommender, OSRMClient
        print("    route_aware_recommender 導入成功")
        
        from simple_llm_filter import SimpleLLMFilter
        print("    simple_llm_filter 導入成功")
        
        import torch
        print(f"    PyTorch 導入成功 (CUDA可用: {torch.cuda.is_available()})")
        
        return True
    except Exception as e:
        print(f"    導入失敗: {e}")
        return False


def test_recommender_creation():
    """測試推薦器創建"""
    print("\n 測試2: 推薦器創建")
    try:
        from route_aware_recommender import create_route_recommender, OSRMClient
        import torch
        
        print("   正在創建推薦器...")
        
        # 創建推薦器
        recommender = create_route_recommender(
            poi_data_path='datasets/meta-California.json.gz',
            model_checkpoint='models/travel_dlrm.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable_spatial_index=True,
            enable_async=True
        )
        
        print("    推薦器創建成功")
        
        # 設置OSRM客戶端
        osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
        recommender.osrm_client = osrm_client
        print("    OSRM客戶端設置成功")
        
        return True, recommender
        
    except Exception as e:
        print(f"    推薦器創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_simple_recommendation(recommender):
    """測試簡單推薦"""
    print("\n 測試3: 生成推薦")
    try:
        print("   正在生成推薦...")
        
        recommendations = recommender.recommend_on_route(
            user_id='1',
            user_history=[],
            start_location=(37.7749, -122.4194),  # 舊金山
            end_location=(37.8199, -122.4783),    # 金門大橋
            top_k=3
        )
        
        if recommendations:
            print(f"    成功生成 {len(recommendations)} 個推薦")
            for i, rec in enumerate(recommendations, 1):
                poi = rec['poi']
                print(f"      {i}. {poi['name']} - 分數: {rec['score']:.3f}")
            return True
        else:
            print("   ️ 沒有生成推薦結果")
            return False
            
    except Exception as e:
        print(f"    推薦生成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_filter():
    """測試LLM過濾器"""
    print("\n 測試4: LLM過濾器")
    try:
        from simple_llm_filter import SimpleLLMFilter
        
        print("   正在創建LLM過濾器...")
        llm_filter = SimpleLLMFilter()
        print("    LLM過濾器創建成功")
        print(f"      端點: {llm_filter.base_url}")
        print(f"      模型: {llm_filter.model}")
        
        # 測試備用過濾邏輯（不調用API）
        test_pois = [
            {'name': '星巴克', 'primary_category': 'Cafe'},
            {'name': 'Purely Storage', 'primary_category': 'Self Storage'}
        ]
        
        print("   測試備用過濾邏輯:")
        for poi in test_pois:
            result = llm_filter._fallback_travel_filter(poi)
            status = " 適合" if result else " 不適合"
            print(f"      {poi['name']}: {status}")
        
        return True
        
    except Exception as e:
        print(f"    LLM過濾器測試失敗: {e}")
        return False


def main():
    """運行所有測試"""
    print("="*60)
    print(" RouteX 系統測試")
    print("="*60)
    
    # 測試1: 導入
    if not test_basic_import():
        print("\n 基本導入失敗，無法繼續測試")
        return
    
    # 測試2: 創建推薦器
    success, recommender = test_recommender_creation()
    if not success:
        print("\n 推薦器創建失敗，無法繼續測試")
        return
    
    # 測試3: 生成推薦
    test_simple_recommendation(recommender)
    
    # 測試4: LLM過濾器
    test_llm_filter()
    
    print("\n" + "="*60)
    print(" 所有測試完成!")
    print("="*60)
    print("\n 下一步:")
    print("   - 運行 quick_start.py 查看完整示例")
    print("   - 運行 demo_complete_usage.py 查看進階功能")


if __name__ == "__main__":
    main()
