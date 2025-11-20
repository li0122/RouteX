"""
測試行程推薦功能
"""

import sys
import json

def test_itinerary_api():
    """測試行程推薦 API"""
    import requests
    
    print("="*60)
    print("測試行程推薦 API")
    print("="*60)
    
    # API 端點
    url = "http://localhost:5000/api/itinerary"
    
    # 測試資料：舊金山旅遊
    payload = {
        "start": [37.7749, -122.4194],  # 舊金山市中心
        "end": [37.8199, -122.4783],     # 金門大橋附近
        "activity_intent": "觀光旅遊，想看著名景點和吃美食",
        "time_budget": 240,  # 4小時
        "top_k": 20,
        "user_id": "test_user",
        "user_history": []
    }
    
    print(f"\n📍 起點: {payload['start']}")
    print(f"📍 終點: {payload['end']}")
    print(f"🎯 需求: {payload['activity_intent']}")
    print(f"⏱️ 時間: {payload['time_budget']} 分鐘")
    
    try:
        print(f"\n🌐 發送請求到 {url}...")
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ 請求成功!")
            print(f"\n{'='*60}")
            print(f"📋 行程摘要")
            print(f"{'='*60}")
            print(f"總景點數: {len(result.get('itinerary', []))}")
            print(f"預計總時間: {result.get('total_duration', 0)} 分鐘")
            print(f"預計總距離: {result.get('total_distance', 0):.2f} 公里")
            print(f"\n{result.get('summary', '無摘要')}")
            
            print(f"\n{'='*60}")
            print(f"🗺️ 行程詳情")
            print(f"{'='*60}")
            
            for item in result.get('itinerary', []):
                poi = item['poi']
                print(f"\n{item['order']}. {poi['name']}")
                print(f"   類別: {poi['category']}")
                print(f"   評分: {poi.get('rating', 'N/A')}")
                print(f"   建議停留: {item['estimated_duration']} 分鐘")
                print(f"   選擇理由: {item['reason']}")
                print(f"   座標: ({poi['latitude']:.6f}, {poi['longitude']:.6f})")
            
            if result.get('tips'):
                print(f"\n{'='*60}")
                print(f"💡 旅遊建議")
                print(f"{'='*60}")
                for tip in result['tips']:
                    print(f"  • {tip}")
            
            # 保存結果
            with open('itinerary_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 完整結果已保存到 itinerary_result.json")
            
        else:
            print(f"\n❌ 請求失敗: {response.status_code}")
            print(f"錯誤訊息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 連接失敗: 請確認 Flask 服務器正在運行")
        print(f"   啟動命令: python web_app.py")
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def test_itinerary_directly():
    """直接測試行程推薦功能（不通過 API）"""
    print("="*60)
    print("直接測試行程推薦功能")
    print("="*60)
    
    try:
        from route_aware_recommender import create_route_aware_recommender
        
        print("\n📦 初始化推薦器...")
        recommender = create_route_aware_recommender(
            model_checkpoint="models/travel_dlrm.pth",
            device="cpu"
        )
        
        print("\n🗺️ 開始行程推薦...")
        result = recommender.recommend_itinerary(
            user_id="test_user",
            user_history=[],
            start_location=(37.7749, -122.4194),  # 舊金山市中心
            end_location=(37.8199, -122.4783),     # 金門大橋附近
            activityIntent="觀光旅遊，想看著名景點和吃美食",
            top_k=20,
            time_budget=240
        )
        
        print(f"\n✅ 行程生成成功!")
        print(f"\n{'='*60}")
        print(f"📋 行程摘要")
        print(f"{'='*60}")
        print(f"總景點數: {len(result.get('itinerary', []))}")
        print(f"預計總時間: {result.get('total_duration', 0)} 分鐘")
        print(f"預計總距離: {result.get('total_distance', 0):.2f} 公里")
        print(f"\n{result.get('summary', '無摘要')}")
        
        print(f"\n{'='*60}")
        print(f"🗺️ 行程詳情")
        print(f"{'='*60}")
        
        for item in result.get('itinerary', []):
            poi = item['poi']
            print(f"\n{item['order']}. {poi.get('name', 'Unknown')}")
            print(f"   類別: {poi.get('primary_category', poi.get('category', 'N/A'))}")
            print(f"   評分: {poi.get('avg_rating', 'N/A')}")
            print(f"   建議停留: {item['estimated_duration']} 分鐘")
            print(f"   選擇理由: {item['reason']}")
        
        if result.get('tips'):
            print(f"\n{'='*60}")
            print(f"💡 旅遊建議")
            print(f"{'='*60}")
            for tip in result['tips']:
                print(f"  • {tip}")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n選擇測試方式:")
    print("1. 測試 API (需要先啟動 web_app.py)")
    print("2. 直接測試功能")
    
    choice = input("\n請選擇 (1/2): ").strip()
    
    if choice == "1":
        test_itinerary_api()
    elif choice == "2":
        test_itinerary_directly()
    else:
        print("無效選擇，執行 API 測試...")
        test_itinerary_api()
