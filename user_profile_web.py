"""
用戶畫像設定 Web 介面
允許用戶設定偏好，並基於畫像生成個性化推薦
"""

from flask import Flask, render_template, request, jsonify
import torch
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from dlrm_model import create_travel_dlrm
from data_processor import POIDataProcessor

app = Flask(__name__)

# 全局變量
model = None
poi_processor = None
device = 'cpu'

def load_model_and_data():
    """載入模型和資料處理器"""
    global model, poi_processor, device
    
    print("正在載入模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 載入 POI 資料
    poi_processor = POIDataProcessor('datasets/meta-California.json.gz')
    poi_processor.load_data()
    poi_processor.preprocess()
    
    # 載入模型
    checkpoint = torch.load('models/travel_dlrm.pth', map_location=device)
    
    if 'poi_vocab_sizes' in checkpoint:
        poi_vocab_sizes = checkpoint['poi_vocab_sizes']
    else:
        poi_vocab_sizes = {
            'category': len(poi_processor.category_encoder),
            'state': len(poi_processor.state_encoder),
            'price_level': 5
        }
    
    model = create_travel_dlrm(
        user_continuous_dim=10,
        poi_continuous_dim=8,
        path_continuous_dim=4,
        user_vocab_sizes={},
        poi_vocab_sizes=poi_vocab_sizes,
        embedding_dim=64,
        bottom_mlp_dims=[256, 128],
        top_mlp_dims=[512, 256, 128],
        dropout=0.2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ 模型載入完成")

def create_user_features(profile: Dict) -> Dict[str, torch.Tensor]:
    """
    根據用戶畫像創建特徵
    
    Args:
        profile: {
            'avg_rating': 用戶平均評分 (1-5)
            'rating_std': 評分標準差 (0-2)
            'num_reviews': 評論數量
            'preferred_categories': 偏好類別列表
            'budget': 預算等級 (1-5)
        }
    
    Returns:
        用戶特徵字典
    """
    # 用戶連續特徵 (10 維)
    user_continuous = torch.tensor([
        profile.get('avg_rating', 4.0),
        profile.get('rating_std', 0.5),
        profile.get('num_reviews', 10),
        profile.get('budget', 3) / 5.0,  # 歸一化
        0, 0, 0, 0, 0, 0  # 其他特徵填充
    ], dtype=torch.float32)
    
    # 用戶類別特徵（當前模型不使用）
    user_categorical = {}
    
    return {
        'user_continuous': user_continuous,
        'user_categorical': user_categorical
    }

def get_recommendations(user_features: Dict, top_k: int = 20, 
                       category_filter: List[str] = None,
                       state_filter: str = None,
                       price_filter: List[int] = None) -> List[Dict]:
    """
    生成推薦並應用過濾條件
    
    Args:
        user_features: 用戶特徵
        top_k: 推薦數量
        category_filter: 類別過濾列表
        state_filter: 州/地區過濾
        price_filter: 價格等級過濾 [min, max]
    
    Returns:
        推薦 POI 列表
    """
    # 應用過濾條件
    candidate_pois = []
    for poi in poi_processor.processed_pois:
        # 類別過濾
        if category_filter:
            if poi.get('primary_category') not in category_filter:
                continue
        
        # 地區過濾
        if state_filter and state_filter != 'all':
            if poi.get('state') != state_filter:
                continue
        
        # 價格過濾
        if price_filter:
            price_level = poi.get('price_level', 0)
            if price_level < price_filter[0] or price_level > price_filter[1]:
                continue
        
        candidate_pois.append(poi['id'])
    
    if len(candidate_pois) == 0:
        return []
    
    # 批次預測
    batch_size = 512
    all_scores = []
    poi_ids = []
    
    with torch.no_grad():
        for i in range(0, len(candidate_pois), batch_size):
            batch_poi_ids = candidate_pois[i:i+batch_size]
            batch_size_actual = len(batch_poi_ids)
            
            # 準備批次特徵
            user_continuous = user_features['user_continuous'].repeat(batch_size_actual, 1).to(device)
            user_categorical = {}
            path_continuous = torch.zeros(batch_size_actual, 4).to(device)
            
            # POI 特徵
            poi_continuous_list = []
            poi_categorical_lists = {'category': [], 'state': [], 'price_level': []}
            
            for poi_id in batch_poi_ids:
                poi_idx = poi_processor.poi_index.get(poi_id)
                if poi_idx is None or poi_idx >= len(poi_processor.processed_pois):
                    poi_continuous_list.append(torch.zeros(8))
                    poi_categorical_lists['category'].append(0)
                    poi_categorical_lists['state'].append(0)
                    poi_categorical_lists['price_level'].append(2)
                else:
                    poi_data = poi_processor.processed_pois[poi_idx]
                    
                    poi_continuous = torch.tensor([
                        poi_data.get('avg_rating', 3.5),
                        poi_data.get('num_reviews', 0),
                        poi_data.get('price_level', 2),
                        poi_data.get('latitude', 0),
                        poi_data.get('longitude', 0),
                        0, 0, 0
                    ], dtype=torch.float32)
                    
                    category_encoded = poi_processor.category_encoder.get(
                        poi_data.get('primary_category', 'Other'),
                        poi_processor.category_encoder.get('Other', 0)
                    )
                    state_encoded = poi_processor.state_encoder.get(
                        poi_data.get('state', 'Unknown'),
                        poi_processor.state_encoder.get('Unknown', 0)
                    )
                    
                    poi_continuous_list.append(poi_continuous)
                    poi_categorical_lists['category'].append(category_encoded)
                    poi_categorical_lists['state'].append(state_encoded)
                    poi_categorical_lists['price_level'].append(min(poi_data.get('price_level', 2), 4))
            
            poi_continuous = torch.stack(poi_continuous_list).to(device)
            poi_categorical = {
                key: torch.tensor(vals, dtype=torch.long).to(device)
                for key, vals in poi_categorical_lists.items()
            }
            
            # 模型預測
            output = model(
                user_continuous,
                user_categorical,
                poi_continuous,
                poi_categorical,
                path_continuous
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
        poi_idx = poi_processor.poi_index.get(poi_id)
        if poi_idx is not None and poi_idx < len(poi_processor.processed_pois):
            poi_data = poi_processor.processed_pois[poi_idx]
            recommendations.append({
                'id': poi_id,
                'name': poi_data.get('name', 'Unknown'),
                'category': poi_data.get('primary_category', 'Other'),
                'rating': poi_data.get('avg_rating', 0),
                'num_reviews': poi_data.get('num_reviews', 0),
                'price_level': poi_data.get('price_level', 0),
                'state': poi_data.get('state', 'Unknown'),
                'address': poi_data.get('address', ''),
                'score': float(score)
            })
    
    return recommendations

@app.route('/')
def index():
    """首頁"""
    # 獲取可用的類別和州
    categories = sorted(poi_processor.category_encoder.keys())
    states = sorted(poi_processor.state_encoder.keys())
    
    return render_template('user_profile.html', 
                         categories=categories,
                         states=states)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """生成推薦"""
    try:
        data = request.json
        
        # 創建用戶特徵
        user_profile = {
            'avg_rating': float(data.get('avg_rating', 4.0)),
            'rating_std': float(data.get('rating_std', 0.5)),
            'num_reviews': int(data.get('num_reviews', 10)),
            'budget': int(data.get('budget', 3))
        }
        
        user_features = create_user_features(user_profile)
        
        # 過濾條件
        category_filter = data.get('categories', [])
        if category_filter and 'all' in category_filter:
            category_filter = None
        
        state_filter = data.get('state', 'all')
        
        price_range = data.get('price_range', [0, 4])
        
        # 生成推薦
        recommendations = get_recommendations(
            user_features,
            top_k=int(data.get('top_k', 20)),
            category_filter=category_filter,
            state_filter=state_filter,
            price_filter=price_range
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/categories')
def get_categories():
    """獲取所有類別"""
    categories = sorted(poi_processor.category_encoder.keys())
    return jsonify({'categories': categories})

@app.route('/api/states')
def get_states():
    """獲取所有州"""
    states = sorted(poi_processor.state_encoder.keys())
    return jsonify({'states': states})

if __name__ == '__main__':
    # 載入模型
    load_model_and_data()
    
    # 啟動服務
    print("\n" + "="*60)
    print("用戶畫像設定 Web 服務")
    print("="*60)
    print("訪問: http://127.0.0.1:5001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
