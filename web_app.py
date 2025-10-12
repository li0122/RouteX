"""
RouteX Web API Server
Flask後端服務器，提供推薦API
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import os
import sys

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
        osrm_client = OSRMClient(server_url="http://router.project-osrm.org")
        print("✅ OSRM客戶端創建成功")
        
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
        
        # 預設不啟用LLM（按需啟用）
        recommender.enable_llm_filter = False
        recommender.llm_filter = None
        
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
        enable_llm = data.get('enable_llm', False)
        
        # 設置LLM過濾器
        if enable_llm and not recommender.enable_llm_filter:
            try:
                recommender.enable_llm_filter = True
                recommender.llm_filter = SimpleLLMFilter()
                print("✅ LLM過濾器已啟用")
            except Exception as e:
                print(f"⚠️ LLM過濾器啟用失敗: {e}")
                recommender.enable_llm_filter = False
        elif not enable_llm and recommender.enable_llm_filter:
            recommender.enable_llm_filter = False
            print("ℹ️ LLM過濾器已禁用")
        
        print(f"\n📍 收到推薦請求:")
        print(f"   起點: {start_location}")
        print(f"   終點: {end_location}")
        print(f"   類別偏好: {categories}")
        print(f"   推薦數量: {top_k}")
        print(f"   LLM過濾: {enable_llm}")
        
        # 構建用戶歷史（可以根據類別偏好構建）
        user_history = []
        if categories:
            # 為每個偏好類別創建虛擬歷史記錄
            for category in categories:
                user_history.append({
                    'category': category,
                    'rating': 5  # 高評分表示偏好
                })
        
        # 調用推薦器
        recommendations = recommender.recommend_on_route(
            user_id='web_user',
            user_history=user_history,
            start_location=tuple(start_location),
            end_location=tuple(end_location),
            top_k=top_k,
            max_detour_ratio=1.3,
            max_extra_duration=900
        )
        
        print(f"生成 {len(recommendations)} 個推薦")
        
        # 格式化返回結果
        result = {
            'success': True,
            'start_location': start_location,
            'end_location': end_location,
            'categories': categories,
            'top_k': top_k,
            'enable_llm': enable_llm,
            'count': len(recommendations),
            'recommendations': format_recommendations(recommendations)
        }
        
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
    """格式化推薦結果"""
    formatted = []
    
    for rec in recommendations:
        poi = rec['poi']
        
        formatted_rec = {
            'poi': {
                'name': poi.get('name', '未知地點'),
                'primary_category': poi.get('primary_category', '未分類'),
                'avg_rating': float(poi.get('avg_rating', 0)) if poi.get('avg_rating') else None,
                'num_reviews': int(poi.get('num_reviews', 0)) if poi.get('num_reviews') else 0,
                'latitude': float(poi.get('latitude', 0)),
                'longitude': float(poi.get('longitude', 0))
            },
            'score': float(rec.get('score', 0)),
            'extra_time_minutes': float(rec.get('extra_time_minutes', 0)),
            'llm_approved': rec.get('llm_approved', False),
            'detour_info': rec.get('detour_info', {}),
            'reasons': rec.get('reasons', [])
        }
        
        formatted.append(formatted_rec)
    
    return formatted


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """獲取可用類別列表"""
    return jsonify({
        'categories': list(CATEGORY_MAP.keys()),
        'category_map': CATEGORY_MAP
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """獲取系統狀態"""
    return jsonify({
        'status': 'running',
        'recommender_initialized': recommender is not None,
        'device': str(recommender.device) if recommender else None,
        'spatial_index_enabled': recommender.spatial_index is not None if recommender else False,
        'llm_filter_available': recommender.enable_llm_filter if recommender else False
    })


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
    print("🚀 RouteX Web Server")
    print("="*60)
    
    init_recommender()
    
    print("\n" + "="*60)
    print("服務器準備就緒!")
    print("="*60)
    print("訪問地址: http://localhost:5000")
    print("API文檔: http://localhost:5000/api/status")
    print("="*60)
    print()
    
    # 啟動Flask服務器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
