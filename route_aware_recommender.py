"""
路徑感知推薦引擎
整合 OSRM 路徑規劃與 DLRM 推薦模型
"""

import torch
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import time

from dlrm_model import TravelDLRM, create_travel_dlrm
from data_processor import POIDataProcessor


class OSRMClient:
    """OSRM 路徑規劃客戶端"""
    
    def __init__(self, server_url: str = "http://router.project-osrm.org"):
        self.server_url = server_url
        self.cache_size = 1000
    
    @lru_cache(maxsize=1000)
    def get_route(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        profile: str = "driving"
    ) -> Optional[Dict]:
        """
        獲取兩點間的路線
        
        Args:
            start: (latitude, longitude)
            end: (latitude, longitude)
            profile: driving, walking, cycling
        
        Returns:
            {
                'distance': 距離(米),
                'duration': 時間(秒),
                'geometry': 路線幾何
            }
        """
        try:
            # OSRM API 格式: longitude,latitude
            url = f"{self.server_url}/route/v1/{profile}/{start[1]},{start[0]};{end[1]},{end[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') == 'Ok' and 'routes' in data:
                route = data['routes'][0]
                return {
                    'distance': route['distance'],  # 米
                    'duration': route['duration'],  # 秒
                    'geometry': route['geometry']
                }
            
            return None
            
        except Exception as e:
            print(f"OSRM 請求失敗: {e}")
            return None
    
    def calculate_detour(
        self,
        start: Tuple[float, float],
        waypoint: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        計算繞道成本
        
        Returns:
            {
                'direct_distance': 直達距離,
                'direct_duration': 直達時間,
                'via_distance': 經過waypoint的距離,
                'via_duration': 經過waypoint的時間,
                'extra_distance': 額外距離,
                'extra_duration': 額外時間,
                'detour_ratio': 繞道比例
            }
        """
        # 直達路線
        direct_route = self.get_route(start, end)
        
        if not direct_route:
            return {
                'direct_distance': 0,
                'direct_duration': 0,
                'via_distance': 0,
                'via_duration': 0,
                'extra_distance': 0,
                'extra_duration': 0,
                'detour_ratio': 0
            }
        
        # 經過waypoint的路線
        route_1 = self.get_route(start, waypoint)
        route_2 = self.get_route(waypoint, end)
        
        if not route_1 or not route_2:
            via_distance = float('inf')
            via_duration = float('inf')
        else:
            via_distance = route_1['distance'] + route_2['distance']
            via_duration = route_1['duration'] + route_2['duration']
        
        extra_distance = max(0, via_distance - direct_route['distance'])
        extra_duration = max(0, via_duration - direct_route['duration'])
        
        detour_ratio = via_distance / direct_route['distance'] if direct_route['distance'] > 0 else float('inf')
        
        return {
            'direct_distance': direct_route['distance'],
            'direct_duration': direct_route['duration'],
            'via_distance': via_distance,
            'via_duration': via_duration,
            'extra_distance': extra_distance,
            'extra_duration': extra_duration,
            'detour_ratio': detour_ratio
        }
    
    def is_poi_on_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        poi: Tuple[float, float],
        max_detour_ratio: float = 1.2,
        max_extra_duration: float = 600  # 10分鐘
    ) -> bool:
        """
        判斷POI是否在合理的路線上
        
        Args:
            start: 起點
            end: 終點
            poi: POI位置
            max_detour_ratio: 最大繞道比例
            max_extra_duration: 最大額外時間(秒)
        
        Returns:
            是否在路線上
        """
        detour = self.calculate_detour(start, poi, end)
        
        return (
            detour['detour_ratio'] <= max_detour_ratio and
            detour['extra_duration'] <= max_extra_duration
        )


class UserPreferenceModel:
    """用戶偏好模型"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.user_profiles = {}
    
    def build_user_profile(
        self, 
        user_id: str, 
        historical_visits: List[Dict]
    ) -> Dict[str, Any]:
        """
        從歷史記錄建立用戶畫像
        
        Args:
            user_id: 用戶ID
            historical_visits: 歷史訪問記錄
        
        Returns:
            用戶畫像
        """
        if not historical_visits:
            return self._default_profile()
        
        # 統計特徵
        ratings = [v.get('rating', 0) for v in historical_visits if v.get('rating')]
        categories = [v.get('category', 'Other') for v in historical_visits if v.get('category')]
        
        from collections import Counter
        category_counts = Counter(categories)
        
        # 偏好類別
        preferred_categories = [cat for cat, _ in category_counts.most_common(5)]
        
        # 平均評分
        avg_rating = np.mean(ratings) if ratings else 3.0
        rating_std = np.std(ratings) if len(ratings) > 1 else 0.5
        
        # 活躍度
        activity_level = len(historical_visits)
        
        profile = {
            'user_id': user_id,
            'avg_rating': avg_rating,
            'rating_std': rating_std,
            'preferred_categories': preferred_categories,
            'activity_level': activity_level,
            'num_visits': len(historical_visits),
            'category_distribution': dict(category_counts)
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def _default_profile(self) -> Dict:
        """默認用戶畫像"""
        return {
            'user_id': 'unknown',
            'avg_rating': 3.5,
            'rating_std': 0.5,
            'preferred_categories': [],
            'activity_level': 0,
            'num_visits': 0,
            'category_distribution': {}
        }
    
    def get_user_features(self, user_id: str) -> np.ndarray:
        """
        獲取用戶特徵向量
        
        Returns:
            (user_continuous_dim,) numpy array
        """
        profile = self.user_profiles.get(user_id, self._default_profile())
        
        features = np.array([
            profile['avg_rating'] / 5.0,  # 標準化
            profile['rating_std'],
            np.log1p(profile['activity_level']),
            np.log1p(profile['num_visits']),
            len(profile['preferred_categories']) / 10.0,
            # 預留特徵
            0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        return features


class RouteAwareRecommender:
    """路徑感知推薦器"""
    
    def __init__(
        self,
        model: TravelDLRM,
        poi_processor: POIDataProcessor,
        osrm_client: Optional[OSRMClient] = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.poi_processor = poi_processor
        self.osrm_client = osrm_client or OSRMClient()
        self.device = torch.device(device)
        self.user_preference_model = UserPreferenceModel()
        
        self.model.to(self.device)
        self.model.eval()
    
    def recommend_on_route(
        self,
        user_id: str,
        user_history: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        candidate_pois: Optional[List[Dict]] = None,
        top_k: int = 10,
        max_detour_ratio: float = 1.3,
        max_extra_duration: float = 900  # 15分鐘
    ) -> List[Dict]:
        """
        在路線上推薦景點
        
        Args:
            user_id: 用戶ID
            user_history: 用戶歷史記錄
            start_location: 起點 (lat, lon)
            end_location: 終點 (lat, lon)
            candidate_pois: 候選POI列表 (None則自動搜索)
            top_k: 返回前K個推薦
            max_detour_ratio: 最大繞道比例
            max_extra_duration: 最大額外時間
        
        Returns:
            推薦結果列表
        """
        # 1. 建立用戶畫像
        user_profile = self.user_preference_model.build_user_profile(
            user_id, user_history
        )
        
        # 2. 獲取候選POI
        if candidate_pois is None:
            # 在路線附近搜索POI
            mid_lat = (start_location[0] + end_location[0]) / 2
            mid_lon = (start_location[1] + end_location[1]) / 2
            candidate_pois = self.poi_processor.get_pois_by_location(
                mid_lat, mid_lon, radius_km=50.0
            )
        
        if not candidate_pois:
            print("警告: 沒有找到候選POI")
            return []
        
        print(f"找到 {len(candidate_pois)} 個候選POI")
        
        # 3. 過濾在路線上的POI
        on_route_pois = []
        for poi in candidate_pois:
            poi_location = (poi['latitude'], poi['longitude'])
            
            # 檢查是否在路線上
            if self.osrm_client.is_poi_on_route(
                start_location, end_location, poi_location,
                max_detour_ratio, max_extra_duration
            ):
                on_route_pois.append(poi)
        
        if not on_route_pois:
            print("警告: 沒有POI在合理的路線上")
            # 降低標準重試
            on_route_pois = candidate_pois[:min(50, len(candidate_pois))]
        
        print(f"過濾後剩餘 {len(on_route_pois)} 個沿途POI")
        
        # 4. 模型評分
        scores = self._score_pois(
            user_profile, 
            on_route_pois,
            start_location,
            end_location
        )
        
        # 5. 排序並返回top-k
        ranked_pois = sorted(
            zip(on_route_pois, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 6. 生成推薦結果
        recommendations = []
        for poi, score in ranked_pois:
            # 計算繞道信息
            poi_location = (poi['latitude'], poi['longitude'])
            detour_info = self.osrm_client.calculate_detour(
                start_location, poi_location, end_location
            )
            
            # 生成推薦理由
            reasons = self._generate_recommendation_reasons(
                poi, user_profile, score, detour_info
            )
            
            recommendations.append({
                'poi': poi,
                'score': float(score),
                'detour_info': detour_info,
                'reasons': reasons,
                'extra_time_minutes': detour_info['extra_duration'] / 60.0
            })
        
        return recommendations
    
    def _score_pois(
        self,
        user_profile: Dict,
        pois: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float]
    ) -> List[float]:
        """使用模型為POI評分"""
        if not pois:
            return []
        
        batch_size = len(pois)
        
        # 準備用戶特徵
        user_continuous = self.user_preference_model.get_user_features(
            user_profile['user_id']
        )
        user_continuous = torch.from_numpy(user_continuous).unsqueeze(0).repeat(batch_size, 1)
        
        user_categorical = {
            # 可以添加用戶類別特徵
        }
        
        # 準備POI特徵
        poi_continuous_list = []
        poi_categorical_dict = {'category': [], 'state': [], 'price_level': []}
        
        for poi in pois:
            encoded = self.poi_processor.encode_poi(poi)
            poi_continuous_list.append(encoded['continuous'])
            
            for key in poi_categorical_dict.keys():
                poi_categorical_dict[key].append(encoded['categorical'].get(key, 0))
        
        poi_continuous = torch.from_numpy(np.array(poi_continuous_list))
        poi_categorical = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in poi_categorical_dict.items()
        }
        
        # 準備路徑特徵
        path_continuous_list = []
        for poi in pois:
            poi_location = (poi['latitude'], poi['longitude'])
            detour = self.osrm_client.calculate_detour(
                start_location, poi_location, end_location
            )
            
            path_features = np.array([
                min(detour['extra_distance'] / 10000.0, 1.0),  # 標準化
                min(detour['extra_duration'] / 3600.0, 1.0),
                min(detour['detour_ratio'] - 1.0, 1.0),
                0.0  # 預留
            ], dtype=np.float32)
            
            path_continuous_list.append(path_features)
        
        path_continuous = torch.from_numpy(np.array(path_continuous_list))
        
        # 移動到設備
        user_continuous = user_continuous.to(self.device)
        poi_continuous = poi_continuous.to(self.device)
        path_continuous = path_continuous.to(self.device)
        
        for key in user_categorical:
            user_categorical[key] = user_categorical[key].to(self.device)
        for key in poi_categorical:
            poi_categorical[key] = poi_categorical[key].to(self.device)
        
        # 模型預測
        with torch.no_grad():
            scores = self.model.predict(
                user_continuous, user_categorical,
                poi_continuous, poi_categorical,
                path_continuous
            )
        
        return scores.cpu().numpy().tolist()
    
    def _generate_recommendation_reasons(
        self,
        poi: Dict,
        user_profile: Dict,
        score: float,
        detour_info: Dict
    ) -> List[str]:
        """生成推薦理由"""
        reasons = []
        
        # 評分高
        if poi.get('avg_rating', 0) >= 4.5:
            reasons.append(f"⭐ 高評分景點 ({poi['avg_rating']:.1f}/5.0)")
        
        # 熱門
        if poi.get('num_reviews', 0) > 100:
            reasons.append(f"🔥 熱門景點 ({poi['num_reviews']} 條評論)")
        
        # 用戶偏好類別
        poi_category = poi.get('primary_category', '')
        if poi_category in user_profile.get('preferred_categories', []):
            reasons.append(f"💡 符合您的偏好 ({poi_category})")
        
        # 繞道時間短
        extra_minutes = detour_info['extra_duration'] / 60.0
        if extra_minutes < 5:
            reasons.append(f"🚗 幾乎不繞路 (僅需額外 {extra_minutes:.0f} 分鐘)")
        elif extra_minutes < 15:
            reasons.append(f"🚗 小幅繞路 (額外 {extra_minutes:.0f} 分鐘)")
        
        # 價格合適
        price_level = poi.get('price_level', 0)
        if price_level <= 2:
            reasons.append("💰 價格實惠")
        
        # 24小時營業
        if poi.get('is_open_24h', False):
            reasons.append("🕐 24小時營業")
        
        # 推薦分數高
        if score > 0.8:
            reasons.append("⭐ 強烈推薦!")
        
        return reasons[:3]  # 最多返回3個理由


def create_route_recommender(
    poi_data_path: str = "datasets/meta-California.json.gz",
    model_checkpoint: Optional[str] = None,
    device: str = 'cpu'
) -> RouteAwareRecommender:
    """
    創建路徑感知推薦器
    
    Args:
        poi_data_path: POI數據路徑
        model_checkpoint: 模型檢查點路徑
        device: 運算設備
    
    Returns:
        RouteAwareRecommender 實例
    """
    print("正在初始化路徑感知推薦器...")
    
    # 載入POI數據
    poi_processor = POIDataProcessor(poi_data_path)
    poi_processor.load_data(max_records=10000)
    poi_processor.preprocess()
    
    # 創建模型
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
        embedding_dim=64
    )
    
    # 載入模型權重
    if model_checkpoint:
        print(f"載入模型權重: {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 創建OSRM客戶端
    osrm_client = OSRMClient()
    
    # 創建推薦器
    recommender = RouteAwareRecommender(
        model=model,
        poi_processor=poi_processor,
        osrm_client=osrm_client,
        device=device
    )
    
    print("✓ 路徑感知推薦器初始化完成!")
    
    return recommender


if __name__ == "__main__":
    print("=== 路徑感知推薦引擎測試 ===\n")
    
    # 測試OSRM客戶端
    osrm = OSRMClient()
    
    # 金門大橋 → 迪士尼樂園
    start = (37.8199, -122.4783)  # 金門大橋
    end = (33.8121, -117.9190)  # 迪士尼樂園
    
    print("測試路徑查詢:")
    route = osrm.get_route(start, end)
    if route:
        print(f"  距離: {route['distance']/1000:.1f} km")
        print(f"  時間: {route['duration']/60:.0f} 分鐘")
    
    # 測試繞道計算
    waypoint = (36.6180, -121.9016)  # 蒙特雷灣水族館
    
    print(f"\n測試繞道計算:")
    detour = osrm.calculate_detour(start, waypoint, end)
    print(f"  直達距離: {detour['direct_distance']/1000:.1f} km")
    print(f"  經過waypoint距離: {detour['via_distance']/1000:.1f} km")
    print(f"  額外距離: {detour['extra_distance']/1000:.1f} km")
    print(f"  繞道比例: {detour['detour_ratio']:.2f}")
    
    print("\n✓ 路徑感知推薦引擎測試完成!")
