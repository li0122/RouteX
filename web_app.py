"""
RouteX Web API Server
Flask後端服務器，提供推薦API
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import os
import sys
import requests

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_aware_recommender import create_route_recommender, OSRMClient
from simple_llm_filter import SimpleLLMFilter

# 創建Flask應用
app = Flask(__name__)
CORS(app)  # 啟用CORS

# 全域變數
recommender = None
osrm_client = None

# 類別映射（中文到英文）
CATEGORY_MAP = {
    'Restaurant': '餐廳',
    'Cafe': '咖啡廳',
    'Museum': '博物館',
    'Park': '公園',
    'Shopping': '購物',
    'Tourist Attraction': '景點',
    'Bar': '酒吧',
    'Entertainment': '娛樂'
}


def init_recommender():
    """初始化推薦器"""
    global recommender, osrm_client
    
    if recommender is not None:
        print("推薦器已初始化")
        return
    
    print("正在初始化推薦系統...")
    
    try:
        # 創建OSRM客戶端
        osrm_client = OSRMClient(server_url="http://140.125.32.60:5000")
        print(" OSRM客戶端創建成功")
        
        # 創建推薦器
        recommender = create_route_recommender(
            poi_data_path='datasets/meta-California.json.gz',
            model_checkpoint='models/travel_dlrm.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable_spatial_index=True,
            enable_async=False  # Web環境使用同步模式
        )
        
        # 設置OSRM客戶端
        recommender.osrm_client = osrm_client
        
        print("推薦系統初始化完成！")
        print(f"   設備: {recommender.device}")
        print(f"   空間索引: {'已啟用' if recommender.spatial_index else '未啟用'}")
        
    except Exception as e:
        print(f"推薦器初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        raise e


@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')


@app.route('/test')
def test():
    """測試頁面"""
    return render_template('test.html')


@app.route('/test_leaflet')
def test_leaflet():
    """Leaflet 地圖測試頁面"""
    return render_template('test_leaflet.html')


@app.route('/profile')
def user_profile():
    """用戶畫像設定頁面"""
    if recommender is None or recommender.poi_processor is None:
        return "推薦系統尚未初始化", 500
    
    # 獲取可用的類別和州
    categories = sorted(recommender.poi_processor.category_encoder.keys())
    states = sorted(recommender.poi_processor.state_encoder.keys())
    
    return render_template('user_profile.html', 
                         categories=categories,
                         states=states)


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """推薦API端點"""
    try:
        # 確保推薦器已初始化
        if recommender is None:
            return jsonify({'error': '推薦系統尚未初始化'}), 500
        
        # 獲取請求數據
        data = request.get_json()
        
        # 驗證必需參數
        if not data:
            return jsonify({'error': '缺少請求數據'}), 400
        
        start_location = data.get('start_location')
        end_location = data.get('end_location')
        
        if not start_location or not end_location:
            return jsonify({'error': '缺少起點或終點座標'}), 400
        
        # 獲取可選參數
        categories = data.get('categories', [])
        top_k = data.get('top_k', 5)
        
        print(f"\n 收到推薦請求:")
        print(f"   起點: {start_location}")
        print(f"   終點: {end_location}")
        
        # 獲取活動意圖
        activity_intent = data.get('activity_intent', '').strip()
        if activity_intent:
            print(f"   活動意圖: {activity_intent} (嚴格審核模式)")
        
        print(f"   類別偏好: {categories}")
        print(f"   推薦數量: {top_k}")
        
        # 構建用戶歷史
        user_history = []
        if activity_intent:
            # 如果有活動意圖，優先使用它（嚴格審核）
            user_history.append({
                'category': activity_intent,
                'rating': 5
            })
        elif categories:
            # 否則使用類別偏好
            for category in categories:
                user_history.append({
                    'category': category,
                    'rating': 5  # 高評分表示偏好
                })
        
        # 調用推薦器
        import time
        start_time = time.time()
        
        try:
            recommendations = recommender.recommend_on_route(
                user_id='web_user',
                user_history=user_history,
                start_location=tuple(start_location),
                end_location=tuple(end_location),
                activityIntent=activity_intent if activity_intent else "旅遊探索",
                top_k=top_k,
                max_detour_ratio=1.3,
                max_extra_duration=900
            )
            
            elapsed = time.time() - start_time
            print(f" 推薦完成: {len(recommendations)} 個，耗時 {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f" 推薦失敗: {e}，耗時 {elapsed:.1f}s")
            raise
        
        # 格式化返回結果
        print(f" 正在格式化 {len(recommendations)} 個推薦結果...")
        
        try:
            formatted_recs = format_recommendations(recommendations)
            print(f" 格式化完成")
        except Exception as e:
            print(f" 格式化失敗: {e}")
            import traceback
            traceback.print_exc()
            # 返回簡化版本
            formatted_recs = [{
                'poi': {
                    'name': rec.get('poi', {}).get('name', '未知'),
                    'primary_category': '未知',
                    'avg_rating': 0,
                    'num_reviews': 0,
                    'latitude': 0,
                    'longitude': 0
                },
                'score': 0,
                'error': '格式化失敗'
            } for rec in recommendations[:top_k]]
        
        result = {
            'success': True,
            'start_location': start_location,
            'end_location': end_location,
            'categories': categories,
            'activity_intent': activity_intent if activity_intent else None,
            'top_k': top_k,
            'count': len(recommendations),
            'recommendations': formatted_recs,
            'processing_time': elapsed
        }
        
        print(f" 返回結果: {len(formatted_recs)} 個推薦")
        return jsonify(result)
        
    except Exception as e:
        print(f"推薦請求失敗: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def format_recommendations(recommendations):
    """格式化推薦結果 - 增強版本"""
    formatted = []
    
    for i, rec in enumerate(recommendations):
        try:
            poi = rec.get('poi', {})
            
            # 安全的數值轉換
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val is not None else default
                except (ValueError, TypeError):
                    return default
            
            def safe_int(val, default=0):
                try:
                    return int(val) if val is not None else default
                except (ValueError, TypeError):
                    return default
            
            formatted_rec = {
                'poi': {
                    'name': str(poi.get('name', '未知地點')),
                    'primary_category': str(poi.get('primary_category', '未分類')),
                    'avg_rating': safe_float(poi.get('avg_rating')),
                    'num_reviews': safe_int(poi.get('num_reviews')),
                    'latitude': safe_float(poi.get('latitude')),
                    'longitude': safe_float(poi.get('longitude'))
                },
                'score': safe_float(rec.get('score')),
                'extra_time_minutes': safe_float(rec.get('extra_time_minutes')),
                'llm_approved': bool(rec.get('llm_approved', False)),
                'detour_info': rec.get('detour_info', {}),
                'reasons': rec.get('reasons', [])
            }
            
            formatted.append(formatted_rec)
            
        except Exception as e:
            print(f"️ 格式化第 {i+1} 個推薦時出錯: {e}")
            # 添加簡化版本
            formatted.append({
                'poi': {
                    'name': '格式化錯誤',
                    'primary_category': '未知',
                    'avg_rating': 0,
                    'num_reviews': 0,
                    'latitude': 0,
                    'longitude': 0
                },
                'error': str(e)
            })
    
    return formatted


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """獲取可用類別列表"""
    if recommender and recommender.poi_processor:
        # 從 POI 處理器獲取實際類別
        actual_categories = list(recommender.poi_processor.category_encoder.keys())
        return jsonify({
            'categories': actual_categories,
            'category_map': CATEGORY_MAP,
            'states': list(recommender.poi_processor.state_encoder.keys())
        })
    else:
        return jsonify({
            'categories': list(CATEGORY_MAP.keys()),
            'category_map': CATEGORY_MAP
        })


@app.route('/api/recommend_by_profile', methods=['POST'])
def recommend_by_profile():
    """
    基於用戶畫像生成推薦
    
    請求格式: {
        "user_profile": {
            "avg_rating": 4.0,
            "rating_std": 0.5,
            "num_reviews": 10,
            "budget": 3
        },
        "filters": {
            "categories": ["Restaurant", "Museum"],
            "state": "California",
            "price_range": [0, 4]
        },
        "top_k": 20
    }
    """
    try:
        if recommender is None:
            return jsonify({'error': '推薦系統尚未初始化'}), 500
        
        data = request.json
        user_profile = data.get('user_profile', {})
        filters = data.get('filters', {})
        top_k = data.get('top_k', 20)
        
        print(f"\n📋 收到用戶畫像推薦請求:")
        print(f"   用戶畫像: {user_profile}")
        print(f"   過濾條件: {filters}")
        print(f"   推薦數量: {top_k}")
        
        # 創建用戶特徵
        import numpy as np
        user_continuous = torch.tensor([
            user_profile.get('avg_rating', 4.0),
            user_profile.get('rating_std', 0.5),
            user_profile.get('num_reviews', 10),
            user_profile.get('budget', 3) / 5.0,
            0, 0, 0, 0, 0, 0
        ], dtype=torch.float32)
        
        user_features = {
            'user_continuous': user_continuous,
            'user_categorical': {}
        }
        
        # 應用過濾條件
        category_filter = filters.get('categories', [])
        if category_filter and 'all' in category_filter:
            category_filter = None
        
        state_filter = filters.get('state', 'all')
        if state_filter == 'all':
            state_filter = None
        
        price_range = filters.get('price_range', [0, 4])
        
        # 篩選候選 POI
        candidate_pois = []
        for poi in recommender.poi_processor.processed_pois:
            # 類別過濾
            if category_filter:
                if poi.get('primary_category') not in category_filter:
                    continue
            
            # 地區過濾
            if state_filter:
                if poi.get('state') != state_filter:
                    continue
            
            # 價格過濾
            price_level = poi.get('price_level', 0)
            if price_level < price_range[0] or price_level > price_range[1]:
                continue
            
            candidate_pois.append(poi['id'])
        
        print(f"   篩選後候選 POI: {len(candidate_pois)}")
        
        if len(candidate_pois) == 0:
            return jsonify({
                'success': True,
                'recommendations': [],
                'count': 0,
                'message': '沒有符合條件的 POI'
            })
        
        # 批次預測
        batch_size = 512
        all_scores = []
        poi_ids = []
        
        with torch.no_grad():
            for i in range(0, len(candidate_pois), batch_size):
                batch_poi_ids = candidate_pois[i:i+batch_size]
                batch_features = prepare_batch_features(
                    user_features, batch_poi_ids, recommender
                )
                
                # 模型預測
                output = recommender.model(
                    batch_features['user_continuous'],
                    batch_features['user_categorical'],
                    batch_features['poi_continuous'],
                    batch_features['poi_categorical'],
                    batch_features['path_continuous']
                )
                
                scores = output['scores'] if isinstance(output, dict) else output
                all_scores.extend(scores.cpu().numpy().flatten().tolist())
                poi_ids.extend(batch_poi_ids)
        
        # 排序並返回 Top-K
        poi_scores = list(zip(poi_ids, all_scores))
        poi_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 構建推薦結果
        recommendations = []
        for poi_id, score in poi_scores[:top_k]:
            poi_idx = recommender.poi_processor.poi_index.get(poi_id)
            if poi_idx is not None and poi_idx < len(recommender.poi_processor.processed_pois):
                poi_data = recommender.poi_processor.processed_pois[poi_idx]
                recommendations.append({
                    'id': poi_id,
                    'name': poi_data.get('name', 'Unknown'),
                    'category': poi_data.get('primary_category', 'Other'),
                    'rating': poi_data.get('avg_rating', 0),
                    'num_reviews': poi_data.get('num_reviews', 0),
                    'price_level': poi_data.get('price_level', 0),
                    'state': poi_data.get('state', 'Unknown'),
                    'address': poi_data.get('address', ''),
                    'latitude': poi_data.get('latitude', 0),
                    'longitude': poi_data.get('longitude', 0),
                    'score': float(score)
                })
        
        print(f"✓ 生成推薦: {len(recommendations)} 個")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        print(f"❌ 用戶畫像推薦失敗: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def prepare_batch_features(user_features, poi_ids, recommender):
    """準備批次特徵用於模型輸入"""
    batch_size = len(poi_ids)
    device = recommender.device
    
    # 複製用戶特徵到批次大小
    batch_features = {
        'user_continuous': user_features['user_continuous'].repeat(batch_size, 1).to(device),
        'user_categorical': {},
        'poi_continuous': [],
        'poi_categorical': {},
        'path_continuous': torch.zeros(batch_size, 4).to(device)
    }
    
    # 複製用戶類別特徵
    for key, val in user_features['user_categorical'].items():
        batch_features['user_categorical'][key] = val.repeat(batch_size).to(device)
    
    # 獲取 POI 特徵
    poi_continuous_list = []
    poi_categorical_lists = {'category': [], 'state': [], 'price_level': []}
    
    for poi_id in poi_ids:
        poi_idx = recommender.poi_processor.poi_index.get(poi_id)
        if poi_idx is None or poi_idx >= len(recommender.poi_processor.processed_pois):
            poi_continuous_list.append(torch.zeros(8))
            poi_categorical_lists['category'].append(0)
            poi_categorical_lists['state'].append(0)
            poi_categorical_lists['price_level'].append(2)
        else:
            poi_data = recommender.poi_processor.processed_pois[poi_idx]
            
            poi_continuous = torch.tensor([
                poi_data.get('avg_rating', 3.5),
                poi_data.get('num_reviews', 0),
                poi_data.get('price_level', 2),
                poi_data.get('latitude', 0),
                poi_data.get('longitude', 0),
                0, 0, 0
            ], dtype=torch.float32)
            
            category_encoded = recommender.poi_processor.category_encoder.get(
                poi_data.get('primary_category', 'Other'),
                recommender.poi_processor.category_encoder.get('Other', 0)
            )
            state_encoded = recommender.poi_processor.state_encoder.get(
                poi_data.get('state', 'Unknown'),
                recommender.poi_processor.state_encoder.get('Unknown', 0)
            )
            
            poi_continuous_list.append(poi_continuous)
            poi_categorical_lists['category'].append(category_encoded)
            poi_categorical_lists['state'].append(state_encoded)
            poi_categorical_lists['price_level'].append(min(poi_data.get('price_level', 2), 4))
    
    # 轉換為張量
    batch_features['poi_continuous'] = torch.stack(poi_continuous_list).to(device)
    batch_features['poi_categorical'] = {
        key: torch.tensor(vals, dtype=torch.long).to(device)
        for key, vals in poi_categorical_lists.items()
    }
    
    return batch_features


@app.route('/api/itinerary', methods=['POST'])
def recommend_itinerary_api():
    """
    推薦完整旅遊行程 API（優化版）
    
    返回格式：單一行程卡片（包含所有景點的完整路線）
    
    請求格式: {
        "start": [lat, lng],
        "end": [lat, lng],
        "activity_intent": "旅遊探索",
        "time_budget": 240,  // 可選，分鐘
        "top_k": 20,  // 可選，DLRM候選數量
        "user_id": "user_123",  // 可選
        "user_history": []  // 可選
    }
    
    返回: {
        "success": true,
        "type": "itinerary",  // 標記為行程類型
        "itinerary": {
            "stops": [...],  // 所有景點（已優化順序）
            "total_duration": 240,
            "total_distance": 150.5,
            "summary": "...",
            "tips": [...],
            "route": {
                "start": [lat, lng],
                "end": [lat, lng],
                "waypoints": [...]  // 中途所有景點
            }
        }
    }
    """
    try:
        data = request.get_json()
        
        # 驗證必要參數
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({'error': '缺少 start 或 end 參數'}), 400
        
        start = tuple(data['start'])
        end = tuple(data['end'])
        activity_intent = data.get('activity_intent', '旅遊探索')
        time_budget = data.get('time_budget')
        top_k = data.get('top_k', 20)
        user_id = data.get('user_id', 'default_user')
        user_history = data.get('user_history', [])
        
        if not recommender:
            return jsonify({'error': '推薦系統未初始化'}), 500
        
        print(f"\n️ 行程推薦請求:")
        print(f"   起點: {start}")
        print(f"   終點: {end}")
        print(f"   活動: {activity_intent}")
        
        # 呼叫行程推薦（含路徑優化）
        import time
        start_time = time.time()
        
        result = recommender.recommend_itinerary(
            user_id=user_id,
            user_history=user_history,
            start_location=start,
            end_location=end,
            activityIntent=activity_intent,
            top_k=top_k,
            time_budget=time_budget
        )
        
        elapsed = time.time() - start_time
        print(f" 行程生成完成，耗時 {elapsed:.1f}s")
        
        # 格式化為單一行程卡片
        stops = []
        waypoints = []
        
        for item in result.get('itinerary', []):
            poi = item.get('poi', {})
            
            # 驗證必要的座標數據
            latitude = poi.get('latitude')
            longitude = poi.get('longitude')
            
            if latitude is None or longitude is None:
                print(f"️ 跳過無效POI: {poi.get('name', 'Unknown')} (缺少座標)")
                continue
            
            stop = {
                'order': item.get('order', 0),
                'name': poi.get('name', 'Unknown'),
                'category': poi.get('primary_category', poi.get('category', 'N/A')),
                'latitude': latitude,
                'longitude': longitude,
                'rating': poi.get('avg_rating', 0),
                'reviews': poi.get('num_reviews', 0),
                'address': poi.get('address', ''),
                'price_level': poi.get('price_level'),
                'reason': item.get('reason', ''),
                'duration': item.get('estimated_duration', 60)
            }
            
            stops.append(stop)
            waypoints.append([latitude, longitude])
        
        # 構建單一行程卡片響應
        itinerary_card = {
            'success': True,
            'type': 'itinerary',  # 標記類型
            'itinerary': {
                'title': f"{activity_intent}行程",
                'stops': stops,
                'total_stops': len(stops),
                'total_duration': result.get('total_duration', 0),
                'total_distance': result.get('total_distance', 0),
                'summary': result.get('summary', ''),
                'tips': result.get('tips', []),
                'route': {
                    'start': list(start),
                    'end': list(end),
                    'waypoints': waypoints,
                    'optimized': result.get('path_optimized', True)
                }
            },
            'processing_time': elapsed
        }
        
        print(f" 返回行程: {len(stops)} 個景點")
        return jsonify(itinerary_card)
        
    except Exception as e:
        print(f" 行程推薦錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """獲取系統狀態"""
    return jsonify({
        'status': 'running',
        'recommender_initialized': recommender is not None,
        'device': str(recommender.device) if recommender else None,
        'spatial_index_enabled': recommender.spatial_index is not None if recommender else False,
        'llm_service_available': recommender.llm_filter is not None if recommender else False
    })


@app.route('/api/route', methods=['POST'])
def get_route():
    """
    獲取 OSRM 路線
    請求格式: {
        "waypoints": [[lat1, lng1], [lat2, lng2], ...],
        "options": {
            "geometries": "geojson",
            "overview": "full",
            "alternatives": false
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'waypoints' not in data:
            return jsonify({'error': '缺少 waypoints 參數'}), 400
        
        waypoints = data['waypoints']
        
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            return jsonify({'error': 'waypoints 必須是至少包含 2 個點的陣列'}), 400
        
        # 驗證每個點的格式
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, list) or len(wp) != 2:
                return jsonify({'error': f'waypoint {i} 格式錯誤，應為 [lat, lng]'}), 400
        
        # 構建 OSRM URL
        coords = ';'.join([f"{wp[1]},{wp[0]}" for wp in waypoints])
        
        # 獲取選項
        options = data.get('options', {})
        geometries = options.get('geometries', 'geojson')
        overview = options.get('overview', 'full')
        alternatives = options.get('alternatives', False)
        
        url = f"http://140.125.32.60:5000/route/v1/driving/{coords}"
        params = {
            'overview': overview,
            'geometries': geometries
        }
        
        if alternatives:
            params['alternatives'] = 'true'
        
        print(f"️ 請求 OSRM 路線: {len(waypoints)} 個點")
        
        # 調用 OSRM API
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        osrm_data = response.json()
        
        if osrm_data.get('code') != 'Ok':
            return jsonify({
                'error': f"OSRM 錯誤: {osrm_data.get('code', 'Unknown')}",
                'message': osrm_data.get('message', '')
            }), 400
        
        # 提取路線資訊
        route = osrm_data.get('routes', [{}])[0]
        
        result = {
            'code': 'Ok',
            'route': {
                'geometry': route.get('geometry'),
                'distance': route.get('distance'),  # 米
                'duration': route.get('duration'),  # 秒
                'legs': route.get('legs', [])
            },
            'waypoints': osrm_data.get('waypoints', [])
        }
        
        print(f" OSRM 路線成功: {route.get('distance', 0)/1000:.1f} km, {route.get('duration', 0)/60:.0f} 分鐘")
        
        return jsonify(result)
        
    except requests.Timeout:
        return jsonify({'error': 'OSRM 請求超時'}), 504
    except requests.RequestException as e:
        print(f" OSRM 請求失敗: {e}")
        return jsonify({'error': f'OSRM 請求失敗: {str(e)}'}), 502
    except Exception as e:
        print(f" 路線獲取錯誤: {e}")
        return jsonify({'error': f'伺服器錯誤: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    """404錯誤處理"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500錯誤處理"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # 初始化推薦器
    print("="*60)
    print(" RouteX Web Server")
    print("="*60)
    
    init_recommender()
    
    print("\n" + "="*60)
    print("服務器準備就緒!")
    print("="*60)
    print("訪問地址: http://localhost:5000")
    print("API文檔: http://localhost:5000/api/status")
    print("="*60)
    print()
    
    # 啟動Flask服務器（正式模式）
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # 正式模式，不會重新加載
        threaded=True
    )
