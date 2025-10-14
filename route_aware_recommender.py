"""
路徑感知推薦引擎
整合 OSRM 路徑規劃與 DLRM 推薦模型
優化版本: 支援異步查詢和空間索引
"""

import torch
import numpy as np
import requests
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import time
import json
import math
from collections import defaultdict

try:
    import aiohttp
    ASYNC_SUPPORTED = True
except ImportError:
    ASYNC_SUPPORTED = False
    print("⚠️ aiohttp未安裝，將使用同步模式")

try:
    from scipy.spatial import cKDTree
    SPATIAL_INDEX_SUPPORTED = True
except ImportError:
    SPATIAL_INDEX_SUPPORTED = False
    print("⚠️ scipy未安裝，將使用線性搜索")

from dlrm_model import TravelDLRM, create_travel_dlrm
from data_processor import POIDataProcessor

try:
    from simple_llm_filter import SimpleLLMFilter
    LLM_FILTER_AVAILABLE = True
except ImportError:
    LLM_FILTER_AVAILABLE = False
    print("⚠️ LLM過濾器不可用，將跳過LLM審核")


class OSRMClient:
    """OSRM 路徑規劃客戶端 - 優化版"""
    
    def __init__(self, server_url: str = "http://router.project-osrm.org"):
        self.server_url = server_url
        self.cache_size = 10000  # 增加緩存大小從1000到10000
        # 使用 Session 連接池提升性能
        self.session = requests.Session()
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate'
        })
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0
        }
    
    @lru_cache(maxsize=10000)  # 增加緩存大小
    def get_route(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        profile: str = "driving"
    ) -> Optional[Dict]:
        """
        獲取兩點間的路線 - 優化版
        
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
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        try:
            # OSRM API 格式: longitude,latitude
            url = f"{self.server_url}/route/v1/{profile}/{start[1]},{start[0]};{end[1]},{end[0]}"
            params = {
                'overview': 'false',  # 減少數據傳輸
                'steps': 'false',     # 不需要詳細步驟
                'alternatives': 'false'  # 不需要替代路線
            }
            
            # 使用重試機制處理暫時性失敗
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.1 * (attempt + 1))  # 指數退避
            
            data = response.json()
            
            if data.get('code') == 'Ok' and 'routes' in data:
                route = data['routes'][0]
                result = {
                    'distance': route['distance'],  # 米
                    'duration': route['duration'],  # 秒
                }
                
                # 更新性能統計
                response_time = time.time() - start_time
                self.performance_stats['avg_response_time'] = (
                    (self.performance_stats['avg_response_time'] * 
                     (self.performance_stats['total_requests'] - 1) + response_time) / 
                    self.performance_stats['total_requests']
                )
                
                return result
            
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
        計算繞道成本 - 優化版
        
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
        try:
            # 直達路線
            direct_route = self.get_route(start, end)
            
            if not direct_route:
                # 如果直達路線失敗，使用距離估算
                direct_distance = self._estimate_distance(start, end) * 1000  # 轉為米
                direct_duration = direct_distance / 15  # 假設15m/s平均速度
                
                direct_route = {
                    'distance': direct_distance,
                    'duration': direct_duration
                }
            
            # 經過waypoint的路線
            route_1 = self.get_route(start, waypoint)
            route_2 = self.get_route(waypoint, end)
            
            if not route_1 or not route_2:
                # 如果繞道路線失敗，使用距離估算
                dist_1 = self._estimate_distance(start, waypoint) * 1000
                dist_2 = self._estimate_distance(waypoint, end) * 1000
                
                via_distance = dist_1 + dist_2
                via_duration = via_distance / 15
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
            
        except Exception as e:
            # 完全失敗時的備用策略
            print(f"   繞道計算失敗: {e}")
            return {
                'direct_distance': 0,
                'direct_duration': 0,
                'via_distance': 0,
                'via_duration': 0,
                'extra_distance': 0,
                'extra_duration': 0,
                'detour_ratio': 0
            }
    
    def _estimate_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """估算兩點間距離(公里)"""
        import math
        R = 6371  # 地球半徑
        
        lat1, lon1, lat2, lon2 = map(math.radians, [start[0], start[1], end[0], end[1]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    async def batch_calculate_detours(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: List[Tuple[float, float]],
        max_concurrent: int = 20
    ) -> List[Optional[Dict]]:
        """
        批量計算繞道成本 - 使用同步 requests（避免 OSRM 封鎖）
        
        注意：OSRM 服務器會封鎖 aiohttp 請求，因此使用同步 requests
        
        Args:
            start: 起點
            end: 終點
            waypoints: 中繼點列表
            max_concurrent: （已棄用，保留參數以兼容）
            
        Returns:
            繞道成本結果列表
        """
        if not waypoints:
            return []
        
        # 使用同步模式（OSRM 會封鎖 aiohttp）
        print(f"   使用同步模式批量計算繞道（避免 OSRM 封鎖 aiohttp）")
        return [self.calculate_detour(start, wp, end) for wp in waypoints]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        return self.performance_stats.copy()


class SpatialIndex:
    """
    空間索引 - 優化POI搜索性能
    從 O(n) 線性搜索優化到 O(log n + k)
    """
    
    def __init__(self, pois: List[Dict]):
        self.pois = pois
        self.index_built = False
        self.kdtree = None
        self.coordinates = None
        self.poi_mapping = {}  # 索引到POI的映射
        
        self._build_index()
        
    def _build_index(self):
        """構建空間索引"""
        if not SPATIAL_INDEX_SUPPORTED:
            print("⚠️ 空間索引不可用，使用線性搜索")
            return
        
        try:
            # 提取有效坐標
            valid_pois = []
            coordinates = []
            
            for i, poi in enumerate(self.pois):
                lat = poi.get('latitude', 0)
                lon = poi.get('longitude', 0)
                
                # 過濾無效坐標
                if lat != 0 and lon != 0 and -90 <= lat <= 90 and -180 <= lon <= 180:
                    coordinates.append([lat, lon])
                    valid_pois.append(poi)
                    self.poi_mapping[len(coordinates) - 1] = poi
            
            if len(coordinates) < 2:
                print("⚠️ 有效POI太少，無法構建空間索引")
                return
            
            # 構建 KD 樹
            self.coordinates = np.array(coordinates)
            self.kdtree = cKDTree(self.coordinates)
            self.index_built = True
            
            print(f"✓ 空間索引構建完成: {len(coordinates):,} 個有效POI")
            
        except Exception as e:
            print(f"空間索引構建失敗: {e}")
            self.index_built = False
    
    def query_by_location(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        max_results: int = 1000
    ) -> List[Dict]:
        """
        按位置查詢POI - O(log n + k)
        
        Args:
            center_lat: 中心緯度
            center_lon: 中心經度
            radius_km: 半徑(公里)
            max_results: 最大結果數
            
        Returns:
            POI列表
        """
        if not self.index_built:
            return self._linear_search(center_lat, center_lon, radius_km, max_results)
        
        try:
            # 轉換半徑到度數 (粗略)
            radius_deg = radius_km / 111.0  # 1度 ≈ 111公里
            
            # KD-tree 球形查詢
            center = np.array([center_lat, center_lon])
            indices = self.kdtree.query_ball_point(center, radius_deg)
            
            # 精確距離過濾和排序
            candidates = []
            for idx in indices:
                if idx in self.poi_mapping:
                    poi = self.poi_mapping[idx]
                    lat, lon = self.coordinates[idx]
                    
                    # 計算精確距離
                    distance = self._haversine_distance(center_lat, center_lon, lat, lon)
                    
                    if distance <= radius_km:
                        candidates.append((poi, distance))
            
            # 按距離排序
            candidates.sort(key=lambda x: x[1])
            
            # 返回結果
            results = [poi for poi, _ in candidates[:max_results]]
            
            print(f"📍 空間索引查詢: {len(results)}/{len(candidates)} POI 在 {radius_km}km 內")
            return results
            
        except Exception as e:
            print(f"空間索引查詢失敗: {e}，回退到線性搜索")
            return self._linear_search(center_lat, center_lon, radius_km, max_results)
    
    def query_by_bbox(
        self,
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
        max_results: int = 1000
    ) -> List[Dict]:
        """按邊界框查詢POI"""
        if not self.index_built:
            return self._linear_bbox_search(min_lat, max_lat, min_lon, max_lon, max_results)
        
        try:
            # 計算中心點和半徑
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            
            # 估算半徑(取更大的邊)
            lat_diff = max_lat - min_lat
            lon_diff = max_lon - min_lon
            radius_deg = max(lat_diff, lon_diff) / 2 * 1.1  # 加倍緩衝
            
            center = np.array([center_lat, center_lon])
            indices = self.kdtree.query_ball_point(center, radius_deg)
            
            # 精確邊界框過濾
            results = []
            for idx in indices:
                if idx in self.poi_mapping:
                    poi = self.poi_mapping[idx]
                    lat, lon = self.coordinates[idx]
                    
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        results.append(poi)
                        
                        if len(results) >= max_results:
                            break
            
            return results
            
        except Exception as e:
            print(f"邊界框查詢失敗: {e}")
            return self._linear_bbox_search(min_lat, max_lat, min_lon, max_lon, max_results)
    
    def _linear_search(
        self, 
        center_lat: float, 
        center_lon: float, 
        radius_km: float,
        max_results: int
    ) -> List[Dict]:
        """線性搜索回退方案"""
        candidates = []
        
        for poi in self.pois:
            lat = poi.get('latitude', 0)
            lon = poi.get('longitude', 0)
            
            if lat != 0 and lon != 0:
                distance = self._haversine_distance(center_lat, center_lon, lat, lon)
                if distance <= radius_km:
                    candidates.append((poi, distance))
        
        # 按距離排序
        candidates.sort(key=lambda x: x[1])
        return [poi for poi, _ in candidates[:max_results]]
    
    def _linear_bbox_search(
        self,
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
        max_results: int
    ) -> List[Dict]:
        """線性邊界框搜索"""
        results = []
        
        for poi in self.pois:
            lat = poi.get('latitude', 0)
            lon = poi.get('longitude', 0)
            
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                results.append(poi)
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """計算兩點間距離(公里)"""
        R = 6371  # 地球半徑
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_index_stats(self) -> Dict[str, Any]:
        """獲取索引統計資訊"""
        return {
            'index_built': self.index_built,
            'total_pois': len(self.pois),
            'indexed_pois': len(self.poi_mapping) if self.index_built else 0,
            'index_type': 'KDTree' if self.index_built else 'Linear'
        }


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
        從歷史記錄建立深度用戶畫像
        
        Args:
            user_id: 用戶ID
            historical_visits: 歷史訪問記錄
        
        Returns:
            詳細用戶畫像
        """
        if not historical_visits:
            return self._default_profile()
        
        # 統計基礎特徵
        ratings = [v.get('rating', 0) for v in historical_visits if v.get('rating')]
        categories = [v.get('category', 'Other') for v in historical_visits if v.get('category')]
        price_levels = [v.get('price_level', 2) for v in historical_visits if v.get('price_level')]
        visit_times = [v.get('visit_time', '') for v in historical_visits if v.get('visit_time')]
        
        from collections import Counter
        category_counts = Counter(categories)
        price_counts = Counter(price_levels)
        
        # 偏好類別分析 (按偏好強度排序)
        total_visits = len(historical_visits)
        category_preferences = {}
        for cat, count in category_counts.items():
            preference_strength = count / total_visits
            category_preferences[cat] = preference_strength
        
        # 按偏好強度排序
        preferred_categories = [cat for cat, _ in 
                              sorted(category_preferences.items(), 
                                    key=lambda x: x[1], reverse=True)]
        
        # 評分習慣分析
        avg_rating = np.mean(ratings) if ratings else 3.5
        rating_std = np.std(ratings) if len(ratings) > 1 else 0.5
        
        # 評分傾向分析
        high_rating_ratio = len([r for r in ratings if r >= 4.0]) / len(ratings) if ratings else 0.5
        rating_generosity = "慷慨" if high_rating_ratio > 0.7 else "嚴格" if high_rating_ratio < 0.3 else "中等"
        
        # 價格偏好分析
        avg_price_level = np.mean(price_levels) if price_levels else 2.0
        price_std = np.std(price_levels) if len(price_levels) > 1 else 1.0
        
        # 價格敏感度分析
        price_variance = price_std / (avg_price_level + 0.1)  # 避免除零
        price_sensitivity = "高" if price_variance < 0.3 else "低" if price_variance > 0.7 else "中等"
        
        # 活躍度與探索性分析
        activity_level = len(historical_visits)
        category_diversity = len(set(categories)) / len(categories) if categories else 0
        exploration_tendency = "高" if category_diversity > 0.6 else "低" if category_diversity < 0.3 else "中等"
        
        # 時間偏好分析 (如果有時間資料)
        time_patterns = {}
        if visit_times:
            # 簡化的時間分析 - 實際應用中可以更詳細
            morning_visits = len([t for t in visit_times if '0' <= t[:2] <= '11'])
            afternoon_visits = len([t for t in visit_times if '12' <= t[:2] <= '17'])
            evening_visits = len([t for t in visit_times if '18' <= t[:2] <= '23'])
            
            total_time_visits = morning_visits + afternoon_visits + evening_visits
            if total_time_visits > 0:
                time_patterns = {
                    'morning_preference': morning_visits / total_time_visits,
                    'afternoon_preference': afternoon_visits / total_time_visits,
                    'evening_preference': evening_visits / total_time_visits
                }
        
        # 建構完整用戶畫像
        profile = {
            # 基礎資訊
            'user_id': user_id,
            'num_visits': len(historical_visits),
            'activity_level': activity_level,
            
            # 評分習慣
            'avg_rating': avg_rating,
            'rating_std': rating_std,
            'rating_generosity': rating_generosity,
            'high_rating_ratio': high_rating_ratio,
            
            # 類別偏好
            'preferred_categories': preferred_categories[:10],  # 前10個偏好
            'category_distribution': dict(category_counts),
            'category_preferences': category_preferences,
            'exploration_tendency': exploration_tendency,
            'category_diversity': category_diversity,
            
            # 價格偏好
            'avg_price_level': avg_price_level,
            'price_std': price_std,
            'price_sensitivity': price_sensitivity,
            'price_distribution': dict(price_counts),
            
            # 時間偏好
            'time_patterns': time_patterns,
            
            # 行為特徵
            'review_frequency': len(ratings) / total_visits if total_visits > 0 else 0,
            'engagement_score': (len(ratings) / total_visits * 0.5 + 
                               category_diversity * 0.3 + 
                               min(activity_level / 20, 1.0) * 0.2) if total_visits > 0 else 0
        }
        
        # 快取用戶畫像
        self.user_profiles[user_id] = profile
        
        # 輸出用戶畫像摘要
        print(f"📊 用戶畫像建立完成:")
        print(f"   偏好類別: {preferred_categories[:3]}")
        print(f"   評分習慣: {avg_rating:.1f}⭐ ({rating_generosity})")
        print(f"   價格偏好: {avg_price_level:.1f}級 ({price_sensitivity}敏感度)")
        print(f"   探索傾向: {exploration_tendency}")
        
        return profile
    
    def _default_profile(self) -> Dict:
        """增強的默認用戶畫像"""
        return {
            # 基礎資訊
            'user_id': 'unknown',
            'num_visits': 0,
            'activity_level': 0,
            
            # 評分習慣
            'avg_rating': 3.5,
            'rating_std': 0.5,
            'rating_generosity': '中等',
            'high_rating_ratio': 0.5,
            
            # 類別偏好
            'preferred_categories': ['Tourist Attraction', 'Restaurant', 'Shopping'],
            'category_distribution': {},
            'category_preferences': {},
            'exploration_tendency': '中等',
            'category_diversity': 0.5,
            
            # 價格偏好
            'avg_price_level': 2.0,
            'price_std': 1.0,
            'price_sensitivity': '中等',
            'price_distribution': {},
            
            # 時間偏好
            'time_patterns': {
                'morning_preference': 0.3,
                'afternoon_preference': 0.4,
                'evening_preference': 0.3
            },
            
            # 行為特徵
            'review_frequency': 0.5,
            'engagement_score': 0.3
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
    """路徑感知推薦器 - 優化版"""
    
    def __init__(
        self,
        model: TravelDLRM,
        poi_processor: POIDataProcessor,
        osrm_client: Optional[OSRMClient] = None,
        device: str = 'cpu',
        enable_spatial_index: bool = True,
        enable_async: bool = True,
        enable_llm_filter: bool = False,
        enable_llm_concurrent: bool = True  # 新增：啟用LLM併發
    ):
        self.model = model
        self.poi_processor = poi_processor
        self.osrm_client = osrm_client or OSRMClient()
        self.device = torch.device(device)
        self.user_preference_model = UserPreferenceModel()
        self.enable_async = enable_async and ASYNC_SUPPORTED
        self.enable_llm_concurrent = enable_llm_concurrent  # 儲存併發設置
        
        # 初始化空間索引
        if enable_spatial_index:
            print("📋 正在構建空間索引...")
            
            # 檢查poi_processor.pois的類型並正確處理
            if hasattr(self.poi_processor, 'pois'):
                if isinstance(self.poi_processor.pois, dict):
                    # 如果是字典，取values
                    all_pois = list(self.poi_processor.pois.values())
                elif isinstance(self.poi_processor.pois, list):
                    # 如果是列表，直接使用
                    all_pois = self.poi_processor.pois
                else:
                    print(f"⚠️ 未知的pois數據類型: {type(self.poi_processor.pois)}")
                    all_pois = []
            else:
                print("⚠️ poi_processor沒有pois屬性")
                all_pois = []
            
            print(f"   找到 {len(all_pois)} 個POI用於空間索引")
            
            if all_pois:
                self.spatial_index = SpatialIndex(all_pois)
            else:
                print("⚠️ 沒有POI數據，禁用空間索引")
                self.spatial_index = None
        else:
            self.spatial_index = None
        
        # 性能統計
        self.performance_stats = {
            'total_recommendations': 0,
            'avg_recommendation_time': 0,
            'spatial_index_hits': 0,
            'async_requests_count': 0
        }
        
        # 初始化LLM過濾器
        self.enable_llm_filter = enable_llm_filter and LLM_FILTER_AVAILABLE
        if self.enable_llm_filter:
            try:
                self.llm_filter = SimpleLLMFilter()
                print(f"✅ LLM過濾器初始化成功")
            except Exception as e:
                print(f"⚠️ LLM過濾器初始化失敗: {e}")
                self.enable_llm_filter = False
                self.llm_filter = None
        else:
            self.llm_filter = None
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 優化版推薦器初始化完成")
        enabled_text = "啟用" if self.spatial_index and self.spatial_index.index_built else "禁用"
        print(f"   - 空間索引: {enabled_text}")
        async_text = "啟用" if self.enable_async else "禁用"
        print(f"   - 異步支持: {async_text}")
        llm_text = "啟用" if self.enable_llm_filter else "禁用"
        print(f"   - LLM過濾器: {llm_text}")
        if self.enable_llm_filter:
            concurrent_text = "啟用" if self.enable_llm_concurrent else "禁用"
            print(f"   - LLM併發: {concurrent_text}")
    
    def recommend_on_route(
        self,
        user_id: str,
        user_history: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        activityIntent: str = "旅遊探索",  # 使用者需求（活動意圖）
        candidate_pois: Optional[List[Dict]] = None,
        top_k: int = 10,
        max_detour_ratio: float = 1.3,
        max_extra_duration: float = 900  # 15分鐘
    ) -> List[Dict]:
        """
        在路線上推薦景點 - 優化版
        
        主要優化:
        1. 地理邊界框過濾
        2. LLM類別智能篩選
        3. 模型評分和LLM審核
        
        Args:
            user_id: 用戶ID
            user_history: 用戶歷史記錄
            start_location: 起點 (lat, lon)
            end_location: 終點 (lat, lon)
            activityIntent: 使用者活動意圖/需求描述
            candidate_pois: 候選POI列表 (None則自動搜索)
            top_k: 返回前K個推薦
            max_detour_ratio: 最大繞道比例 (已棄用)
            max_extra_duration: 最大額外時間 (已棄用)
        
        Returns:
            推薦結果列表
        """
        start_time = time.time()
        self.performance_stats['total_recommendations'] += 1
        
        print(f"🎯 開始路線推薦: {start_location} → {end_location}")
        
        # 1. 建立用戶畫像
        print("👤 步驟1: 建立用戶畫像...")
        user_profile = self.user_preference_model.build_user_profile(
            user_id, user_history
        )
        
        # 2. 獲取所有候選POI（不進行空間索引和預過濾，使篩選更全面）
        if candidate_pois is None:
            # 直接使用所有POI
            if hasattr(self.poi_processor, 'pois'):
                if isinstance(self.poi_processor.pois, dict):
                    candidate_pois = list(self.poi_processor.pois.values())
                elif isinstance(self.poi_processor.pois, list):
                    candidate_pois = self.poi_processor.pois
                else:
                    print(f"⚠️ 未知的pois數據類型: {type(self.poi_processor.pois)}")
                    candidate_pois = []
            else:
                print("⚠️ poi_processor沒有pois屬性")
                candidate_pois = []
        
        print(f"📊 使用全部 {len(candidate_pois)} 個POI進行篩選（未進行空間和預過濾）")
        
        if not candidate_pois:
            print("⚠️ 沒有找到候選POI")
            return []
        
        # 2.5. 地理邊界框過濾（在路線過濾前）
        print("📦 步驟2.5: 地理邊界框過濾...")
        bbox_start = time.time()
        
        filtered_pois = self._filter_by_bounding_box(
            candidate_pois, start_location, end_location, padding_ratio=0.1
        )
        
        bbox_time = time.time() - bbox_start
        filter_rate = (1 - len(filtered_pois) / len(candidate_pois)) * 100 if candidate_pois else 0
        
        print(f"   邊界框: 緯度 [{min(start_location[0], end_location[0]):.6f}, {max(start_location[0], end_location[0]):.6f}]")
        print(f"           經度 [{min(start_location[1], end_location[1]):.6f}, {max(start_location[1], end_location[1]):.6f}]")
        print(f"   矩形內POI: {len(filtered_pois)} 個（過濾掉 {len(candidate_pois) - len(filtered_pois)} 個，{filter_rate:.1f}%）")
        print(f"   耗時: {bbox_time:.3f}s")
        
        if not filtered_pois:
            print("⚠️ 地理邊界框內沒有POI，嘗試放寬範圍...")
            # 如果過濾後沒有POI，放寬邊界框
            filtered_pois = self._filter_by_bounding_box(
                candidate_pois, start_location, end_location, padding_ratio=0.5
            )
            print(f"   放寬後POI: {len(filtered_pois)} 個")
            
            if not filtered_pois:
                print("⚠️ 即使放寬邊界框仍沒有POI")
                return []
        
        # 3. LLM類別篩選（取代OSRM路線過濾）
        print("步驟3: LLM智能類別篩選...")
        llm_start = time.time()
        
        # 提取所有唯一類別
        all_categories = list(set([poi.get('primary_category', '') or poi.get('category', 'Unknown') 
                                   for poi in filtered_pois if poi.get('primary_category') or poi.get('category')]))
        all_categories = [cat for cat in all_categories if cat and cat != 'Unknown']
        
        print(f"   邊界框內共有 {len(all_categories)} 個不同類別")
        print(f"   使用者需求: {activityIntent}")
        
        # 使用LLM篩選符合需求的類別
        selected_categories = self._llm_filter_categories(activityIntent, all_categories)
        
        llm_time = time.time() - llm_start
        print(f"   LLM篩選結果: {len(selected_categories)} 個類別")
        print(f"   符合類別: {', '.join(selected_categories[:10])}{'...' if len(selected_categories) > 10 else ''}")
        print(f"   耗時: {llm_time:.3f}s")
        
        if not selected_categories:
            print("⚠️ 沒有符合的類別")
            return []
        
        # 4. 根據篩選的類別更新POI列表
        category_filtered_pois = [
            poi for poi in filtered_pois 
            if (poi.get('primary_category', '') in selected_categories or 
                poi.get('category', '') in selected_categories)
        ]
        
        print(f"   類別過濾後: {len(category_filtered_pois)} 個POI")
        
        if not category_filtered_pois:
            print("⚠️ 沒有POI匹配篩選的類別")
            return []
        
        # 5. 模型評分
        print("🧠 步驟4: 模型評分...")
        inference_start = time.time()
        
        scores = self._score_pois(
            user_profile, category_filtered_pois, start_location, end_location
        )
        
        inference_time = time.time() - inference_start
        print(f"   模型評分完成 (耗時: {inference_time:.3f}s)")
        
        # 6. 計算 OSRM 繞道信息（針對所有評分後的 POI）
        print("🚗 步驟5: 計算繞道信息...")
        osrm_start = time.time()
        
        # 提取 POI 位置
        poi_locations = [(poi['latitude'], poi['longitude']) for poi in category_filtered_pois]
        
        # 使用線程池並發計算（顯著提升速度）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 根據POI數量動態調整並發數
        max_workers = min(20, max(5, len(poi_locations) // 2))
        print(f"   使用 {max_workers} 個並發線程")
        
        detours = [None] * len(poi_locations)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_idx = {
                executor.submit(
                    self.osrm_client.calculate_detour,
                    start_location, poi_loc, end_location
                ): idx
                for idx, poi_loc in enumerate(poi_locations)
            }
            
            # 收集結果
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    detours[idx] = future.result()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"   進度: {completed}/{len(poi_locations)}")
                except Exception as e:
                    print(f"   POI {idx} 計算失敗: {e}")
                    detours[idx] = None
        
        osrm_time = time.time() - osrm_start
        valid_detours = [d for d in detours if d and d.get('detour_ratio', 0) > 0]
        print(f"   繞道計算完成: {len(valid_detours)}/{len(detours)} 個有效 (耗時: {osrm_time:.3f}s)")
        if osrm_time > 0:
            print(f"   平均速度: {len(detours)/osrm_time:.1f} POI/秒 (並發模式)")
        
        # 7. 生成推薦結果
        recommendations = self._generate_recommendations(
            category_filtered_pois, scores, detours, top_k, user_profile, user_history,
            start_location, end_location
        )
        
        # 更新性能統計
        total_time = time.time() - start_time
        self._update_performance_stats(total_time)
        
        print(f"\n✅ 推薦完成! 總耗時: {total_time:.3f}s")
        print(f"   最終推薦: {len(recommendations)} 個")
        
        return recommendations
    
    def _llm_filter_categories(
        self,
        activityIntent: str,
        all_categories: List[str]
    ) -> List[str]:
        """
        使用LLM根據使用者需求篩選符合的商店類別
        
        Args:
            activityIntent: 使用者活動意圖/需求描述
            all_categories: 所有商店類別列表
        
        Returns:
            符合需求的類別列表
        """
        if not self.enable_llm_filter or not self.llm_filter:
            print("⚠️ LLM過濾器不可用，返回所有類別")
            return all_categories
        
        # 構建prompt
        categories_str = ", ".join(all_categories[:100])  # 限制類別數量避免過長
        if len(all_categories) > 100:
            categories_str += f" ... (共 {len(all_categories)} 個類別)"
        
        prompt = f"""You are a travel recommendation assistant.

User Needs: {activityIntent}

Available Categories:
{categories_str}

Task: Based on the user's needs, select ONLY the relevant categories from the list above that would be useful for their trip.

Rules:
1. Only return categories that directly match the user's needs
2. Return categories as a comma-separated list
3. Use the EXACT category names from the list
4. If user needs are general (like "travel" or "tourism"), include tourist attractions, restaurants, hotels, shopping, etc.
5. If user needs are specific (like "food" or "dining"), only include restaurants and food-related categories

Output Format:
Category1, Category2, Category3, ...

Do NOT include explanations, just return the comma-separated category list."""
        
        try:
            print("   調用LLM篩選類別...")
            response = self.llm_filter._call_llm(prompt)
            
            if not response:
                print("⚠️ LLM調用失敗，返回所有類別")
                return all_categories
            
            # 解析LLM輸出
            selected_categories = [cat.strip() for cat in response.split(',') if cat.strip()]
            
            # 驗證類別是否在原始列表中
            valid_categories = [cat for cat in selected_categories if cat in all_categories]
            
            if not valid_categories:
                print("⚠️ LLM返回的類別無效，使用所有類別")
                return all_categories
            
            return valid_categories
            
        except Exception as e:
            print(f"⚠️ LLM類別篩選失敗: {e}")
            return all_categories
    
    def _filter_by_bounding_box(
        self,
        pois: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        padding_ratio: float = 0.1
    ) -> List[Dict]:
        """
        地理邊界框過濾 - 只保留在起點和終點構成的矩形範圍內的POI
        
        Args:
            pois: 候選POI列表
            start_location: 起點 (lat, lng)
            end_location: 終點 (lat, lng)
            padding_ratio: 邊界框擴展比例（默認10%），避免邊緣POI被過濾
        
        Returns:
            在矩形範圍內的POI列表
        """
        if not pois:
            return []
        
        # 計算矩形邊界
        min_lat = min(start_location[0], end_location[0])
        max_lat = max(start_location[0], end_location[0])
        min_lng = min(start_location[1], end_location[1])
        max_lng = max(start_location[1], end_location[1])
        
        # 計算邊界框尺寸
        lat_range = max_lat - min_lat
        lng_range = max_lng - min_lng
        
        # 添加padding避免過於嚴格（擴展邊界框）
        lat_padding = lat_range * padding_ratio
        lng_padding = lng_range * padding_ratio
        
        min_lat -= lat_padding
        max_lat += lat_padding
        min_lng -= lng_padding
        max_lng += lng_padding
        
        # 過濾POI
        filtered = []
        for poi in pois:
            lat = poi.get('latitude')
            lng = poi.get('longitude')
            
            if lat is None or lng is None:
                continue
            
            # 檢查是否在矩形範圍內
            if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
                filtered.append(poi)
        
        return filtered
    
    def _spatial_search_candidates(
        self,
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        radius_km: Optional[float] = None
    ) -> List[Dict]:
        """
        空間索引搜索候選POI - 智能半徑調整版
        
        根據路線長度動態調整搜索半徑：
        - 短途(<20km): 半徑 20km
        - 中途(20-50km): 半徑 40km
        - 長途(50-100km): 半徑 60km
        - 超長途(>100km): 半徑 80km
        """
        
        # 計算路線直線距離
        route_distance = self._haversine_distance(
            start_location[0], start_location[1],
            end_location[0], end_location[1]
        )
        
        # 根據路線長度動態調整搜索半徑
        if radius_km is None:
            if route_distance < 20:
                radius_km = 20.0  # 市區短途
            elif route_distance < 50:
                radius_km = 40.0  # 城市間中途
            elif route_distance < 100:
                radius_km = 60.0  # 長途
            elif route_distance < 200:
                radius_km = 80.0  # 超長途
            else:
                # 極長途（如跨州）：使用更大半徑或路線長度的 20%
                radius_km = min(120.0, route_distance * 0.2)
        
        print(f"   路線直線距離: {route_distance:.1f} km")
        print(f"   搜索半徑: {radius_km:.1f} km")
        
        if self.spatial_index and self.spatial_index.index_built:
            # 使用空間索引
            mid_lat = (start_location[0] + end_location[0]) / 2
            mid_lon = (start_location[1] + end_location[1]) / 2
            
            candidates = self.spatial_index.query_by_location(
                mid_lat, mid_lon, radius_km, max_results=500  # 增加最大結果數
            )
            
            self.performance_stats['spatial_index_hits'] += 1
            return candidates
        else:
            # 回退到原始方法
            mid_lat = (start_location[0] + end_location[0]) / 2
            mid_lon = (start_location[1] + end_location[1]) / 2
            return self.poi_processor.get_pois_by_location(
                mid_lat, mid_lon, radius_km=radius_km
            )
    
    def _intelligent_prefilter(
        self,
        candidates: List[Dict],
        user_history: List[Dict],
        max_candidates: int = 200
    ) -> List[Dict]:
        """
        智能預過濾候選POI - 改進版
        
        評分標準 (動態權重):
        - 個人化類別偏好: 25-35%
        - 評分質量: 20-30%  
        - 熱門度與可信度: 15-25%
        - 價格匹配度: 10-15%
        - 營業時間便利性: 5-10%
        - 特色標籤匹配: 5-10%
        """
        if len(candidates) <= max_candidates:
            return candidates
        
        # 分析用戶歷史行為模式
        user_categories = [item.get('category', 'Other') for item in user_history]
        user_ratings = [item.get('rating', 3.0) for item in user_history if item.get('rating')]
        user_price_levels = [item.get('price_level', 2) for item in user_history if item.get('price_level')]
        
        # 建立用戶偏好模型
        category_scores = {}
        for category in user_categories:
            category_scores[category] = category_scores.get(category, 0) + 1
        
        total_history = len(user_history) or 1
        for category in category_scores:
            category_scores[category] /= total_history
        
        # 用戶評分習慣
        avg_user_rating = np.mean(user_ratings) if user_ratings else 3.5
        user_rating_std = np.std(user_ratings) if len(user_ratings) > 1 else 0.5
        
        # 用戶價格偏好
        preferred_price_level = np.mean(user_price_levels) if user_price_levels else 2.0
        
        # 計算每個POI的預過濾分數
        scored_pois = []
        for poi in candidates:
            score = 0.0
            
            # 1. 個人化類別偏好 (25-35%)
            poi_category = poi.get('primary_category', 'Other')
            category_preference = category_scores.get(poi_category, 0.1)
            # 動態權重：類別偏好越強，權重越高
            category_weight = 0.25 + (category_preference * 0.1)
            score += category_preference * category_weight
            
            # 2. 評分質量 (20-30%)
            rating = poi.get('avg_rating', 3.0)
            num_reviews = poi.get('num_reviews', 0)
            
            # 考慮評分可信度（評論數越多越可信）
            rating_confidence = min(num_reviews / 50.0, 1.0)
            adjusted_rating = rating * rating_confidence + 3.0 * (1 - rating_confidence)
            
            # 與用戶評分習慣匹配度
            rating_match = 1.0 - abs(adjusted_rating - avg_user_rating) / 5.0
            rating_weight = 0.20 + (rating_confidence * 0.1)
            score += (adjusted_rating / 5.0 * 0.7 + rating_match * 0.3) * rating_weight
            
            # 3. 熱門度與可信度 (15-25%)
            popularity = min(num_reviews / 200.0, 1.0)
            # 避免過度偏向熱門，平衡新穎性
            balanced_popularity = popularity * 0.8 + 0.2
            popularity_weight = 0.15 + (popularity * 0.1)
            score += balanced_popularity * popularity_weight
            
            # 4. 價格匹配度 (10-15%)
            price_level = poi.get('price_level', 2)
            price_match = 1.0 - abs(price_level - preferred_price_level) / 4.0
            price_affordability = (4 - price_level) / 4.0
            price_score = price_match * 0.6 + price_affordability * 0.4
            score += price_score * 0.12
            
            # 5. 營業時間便利性 (5-10%)
            time_convenience = 0.5  # 預設值
            if poi.get('is_open_24h', False):
                time_convenience = 1.0
            elif poi.get('opening_hours'):  # 有營業時間資訊
                time_convenience = 0.8
            score += time_convenience * 0.07
            
            # 6. 特色標籤匹配 (5-10%)
            special_bonus = 0.0
            if poi.get('has_photos', False):
                special_bonus += 0.2
            if poi.get('has_website', False):
                special_bonus += 0.1
            if poi.get('wheelchair_accessible', False):
                special_bonus += 0.1
            if poi.get('good_for_groups', False):
                special_bonus += 0.1
            
            score += min(special_bonus, 1.0) * 0.06
            
            # 7. 多樣性加分 (避免同質化推薦)
            diversity_bonus = 0.0
            if category_preference < 0.3:  # 對於非主要偏好類別給予額外機會
                diversity_bonus = 0.1
            score += diversity_bonus * 0.05
            
            scored_pois.append((poi, score))
        
        # 按分數排序並返回前max_candidates個
        scored_pois.sort(key=lambda x: x[1], reverse=True)
        
        # 加入多樣性過濾，避免同類別過度集中
        diversified_pois = []
        category_counts = {}
        max_per_category = max(3, max_candidates // 5)  # 每個類別最多佔20%
        
        for poi, poi_score in scored_pois:
            poi_category = poi.get('primary_category', 'Other')
            if category_counts.get(poi_category, 0) < max_per_category:
                diversified_pois.append(poi)
                category_counts[poi_category] = category_counts.get(poi_category, 0) + 1
                
                if len(diversified_pois) >= max_candidates:
                    break
        
        # 如果多樣性過濾後數量不足，補充高分POI
        if len(diversified_pois) < max_candidates:
            remaining_pois = [poi for poi, _ in scored_pois if poi not in diversified_pois]
            diversified_pois.extend(remaining_pois[:max_candidates - len(diversified_pois)])
        
        print(f"   智能預過濾: {len(candidates)} → {len(diversified_pois)} (減少 {(1-len(diversified_pois)/len(candidates))*100:.1f}%)")
        print(f"   類別分佈: {[(cat, count) for cat, count in category_counts.items() if count > 0]}")
        
        return diversified_pois
    
    async def _async_route_recommendation(
        self,
        user_profile: Dict,
        filtered_pois: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        top_k: int,
        max_detour_ratio: float,
        max_extra_duration: float,
        start_time: float,
        user_history: List[Dict] = None
    ) -> List[Dict]:
        """異步路線推薦流程（注意：OSRM 使用同步以避免封鎖）"""
        
        print("🚀 步驟4: 路線過濾...")
        osrm_start = time.time()
        
        # 提取POI位置
        poi_locations = [(poi['latitude'], poi['longitude']) for poi in filtered_pois]
        
        # 批量計算繞道成本（使用同步模式避免 OSRM 封鎖）
        detour_results = self.osrm_client.batch_calculate_detours(
            start_location, end_location, poi_locations, max_concurrent=20
        )
        
        self.performance_stats['async_requests_count'] += 1
        
        # 過濾有效結果
        valid_pois = []
        valid_detours = []
        
        for poi, detour in zip(filtered_pois, detour_results):
            if (detour and 
                detour['detour_ratio'] <= max_detour_ratio and 
                detour['extra_duration'] <= max_extra_duration):
                valid_pois.append(poi)
                valid_detours.append(detour)
        
        osrm_time = time.time() - osrm_start
        print(f"   路線過濾完成: {len(valid_pois)} 個有效POI (耗時: {osrm_time:.3f}s)")
        
        if not valid_pois:
            print("⚠️ 沒有POI滿足路線約束")
            return []
        
        # 模型評分
        print("🧠 步驟5: 模型評分...")
        inference_start = time.time()
        
        scores = self._score_pois(
            user_profile, valid_pois, start_location, end_location
        )
        
        inference_time = time.time() - inference_start
        print(f"   模型評分完成 (耗時: {inference_time:.3f}s)")
        
        # 生成推薦結果
        recommendations = self._generate_recommendations(
            valid_pois, scores, valid_detours, top_k, user_profile, user_history,
            start_location, end_location
        )
        
        # 更新性能統計
        total_time = time.time() - start_time
        self._update_performance_stats(total_time)
        
        print(f"\n✅ 推薦完成! 總耗時: {total_time:.3f}s")
        print(f"   最終推薦: {len(recommendations)} 個")
        
        return recommendations
    
    def _sync_route_recommendation(
        self,
        user_profile: Dict,
        filtered_pois: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        top_k: int,
        max_detour_ratio: float,
        max_extra_duration: float,
        start_time: float,
        user_history: List[Dict] = None
    ) -> List[Dict]:
        """同步路線推薦流程 (回退模式) - 優化版"""
        
        print(f"🐢 步驟4: 同步路線過濾 (快速模式)...")
        osrm_start = time.time()
        
        valid_pois = []
        valid_detours = []
        failed_requests = 0
        
        # 先測試直達路線
        print(f"   測試直達路線: {start_location} → {end_location}")
        direct_route = self.osrm_client.get_route(start_location, end_location)
        
        if not direct_route:
            print(f"   ⚠️ 直達路線查詢失敗，使用降級策略")
            # 降級策略: 使用距離估算
            return self._fallback_distance_based_recommendation(
                user_profile, filtered_pois, start_location, end_location, top_k
            )
        
        print(f"   直達路線: {direct_route['distance']/1000:.1f}km, {direct_route['duration']/60:.1f}分鐘")
        
        # 分批處理POI以提高效率
        batch_size = 5  # 每批蔄5個POI
        total_batches = (len(filtered_pois) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(filtered_pois))
            batch_pois = filtered_pois[batch_start:batch_end]
            
            if batch_idx % 5 == 0:  # 每5批報告一次進度
                print(f"   處理批次 {batch_idx+1}/{total_batches}...")
            
            for poi in batch_pois:
                poi_location = (poi['latitude'], poi['longitude'])
                
                try:
                    detour = self.osrm_client.calculate_detour(
                        start_location, poi_location, end_location
                    )
                    
                    # 放寬約束以提高成功率
                    relaxed_detour_ratio = max_detour_ratio * 1.5  # 1.3 -> 1.95
                    relaxed_extra_duration = max_extra_duration * 2  # 900s -> 1800s
                    
                    if (detour['detour_ratio'] <= relaxed_detour_ratio and 
                        detour['extra_duration'] <= relaxed_extra_duration and
                        detour['detour_ratio'] > 0):  # 確保有效數值
                        valid_pois.append(poi)
                        valid_detours.append(detour)
                    
                except Exception as e:
                    failed_requests += 1
                    if failed_requests <= 3:  # 只顯示前3個錯誤
                        print(f"   OSRM查詢失敗: {e}")
                    continue
        
        osrm_time = time.time() - osrm_start
        print(f"   路線過濾完成: {len(valid_pois)} 個有效POI (耗時: {osrm_time:.3f}s)")
        
        if failed_requests > 0:
            print(f"   ⚠️ 失敗查詢: {failed_requests} 個")
        
        if not valid_pois:
            print(f"   ⚠️ 沒有POI通過路線篩選，使用備用策略")
            # 備用策略: 按距離推薦
            return self._fallback_distance_based_recommendation(
                user_profile, filtered_pois, start_location, end_location, top_k
            )
        
        # 模型評分
        print(f"🧠 步驟5: 模型評分...")
        scores = self._score_pois(
            user_profile, valid_pois, start_location, end_location
        )
        
        # 生成推薦結果
        recommendations = self._generate_recommendations(
            valid_pois, scores, valid_detours, top_k, user_profile, user_history,
            start_location, end_location
        )
        
        # 更新性能統計
        total_time = time.time() - start_time
        self._update_performance_stats(total_time)
        
        print(f"\n✅ 推薦完成! 總耗時: {total_time:.3f}s")
        return recommendations
    
    def _fallback_distance_based_recommendation(
        self,
        user_profile: Dict,
        pois: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        top_k: int
    ) -> List[Dict]:
        """
        備用策略: 基於距離的推薦
        當OSRM失敗時使用
        """
        print(f"   使用備用策略: 基於距離的推薦")
        
        # 計算路線中點
        mid_lat = (start_location[0] + end_location[0]) / 2
        mid_lon = (start_location[1] + end_location[1]) / 2
        
        # 計算各POI到路線中點的距離
        poi_distances = []
        for poi in pois:
            distance = self._haversine_distance(
                mid_lat, mid_lon, poi['latitude'], poi['longitude']
            )
            poi_distances.append((poi, distance))
        
        # 按距離排序，取最近的
        poi_distances.sort(key=lambda x: x[1])
        
        # 獲取前top_k個
        selected_pois = [poi for poi, _ in poi_distances[:top_k * 2]]  # 多選一些用於評分
        
        if not selected_pois:
            return []
        
        # 模型評分
        scores = self._score_pois(
            user_profile, selected_pois, start_location, end_location
        )
        
        # 生成模擬繞道信息
        mock_detours = []
        for poi, distance in poi_distances[:len(selected_pois)]:
            mock_detours.append({
                'direct_distance': 500000,  # 500km 模擬
                'direct_duration': 18000,   # 5小時模擬
                'via_distance': 500000 + distance * 1000,
                'via_duration': 18000 + distance * 60,
                'extra_distance': distance * 1000,
                'extra_duration': distance * 60,
                'detour_ratio': 1.0 + (distance / 500)
            })
        
        # 生成推薦結果
        recommendations = self._generate_recommendations(
            selected_pois, scores, mock_detours, top_k, {'preferred_categories': []}, None,
            start_location, end_location
        )
        
        print(f"   備用策略生成 {len(recommendations)} 個推薦")
        return recommendations
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """計算兩點間距離(公里)"""
        import math
        R = 6371  # 地球半徑
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _point_to_line_distance(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> float:
        """
        計算點到線段的最短距離 (公里)
        
        Args:
            point: POI 坐標 (lat, lng)
            line_start: 起點坐標 (lat, lng)
            line_end: 終點坐標 (lat, lng)
        
        Returns:
            POI 到直線的垂直距離 (公里)
        """
        import math
        
        # 轉換為弧度
        lat1, lon1 = math.radians(line_start[0]), math.radians(line_start[1])
        lat2, lon2 = math.radians(line_end[0]), math.radians(line_end[1])
        lat3, lon3 = math.radians(point[0]), math.radians(point[1])
        
        # 計算線段長度
        line_length = self._haversine_distance(
            line_start[0], line_start[1], line_end[0], line_end[1]
        )
        
        if line_length < 0.001:  # 起點和終點太接近
            return self._haversine_distance(
                point[0], point[1], line_start[0], line_start[1]
            )
        
        # 使用向量投影計算點到線段的距離
        # 計算向量
        dx = lon2 - lon1
        dy = lat2 - lat1
        
        # 計算投影參數 t
        t = ((lon3 - lon1) * dx + (lat3 - lat1) * dy) / (dx * dx + dy * dy)
        
        # 限制 t 在 [0, 1] 範圍內（確保投影點在線段上）
        t = max(0, min(1, t))
        
        # 計算投影點坐標
        proj_lat = math.degrees(lat1 + t * dy)
        proj_lon = math.degrees(lon1 + t * dx)
        
        # 計算 POI 到投影點的距離
        perpendicular_dist = self._haversine_distance(
            point[0], point[1], proj_lat, proj_lon
        )
        
        return perpendicular_dist
    
    def _generate_recommendations(
        self,
        pois: List[Dict],
        scores: List[float],
        detours: List[Dict],
        top_k: int,
        user_profile: Dict = None,
        user_history: List[Dict] = None,
        start_location: Tuple[float, float] = None,
        end_location: Tuple[float, float] = None
    ) -> List[Dict]:
        """生成推薦結果 - 包含LLM審核"""
        
        # 如果沒有提供 detours，創建空的 detour 信息
        if detours is None:
            detours = [{
                'direct_distance': 0,
                'direct_duration': 0,
                'via_distance': 0,
                'via_duration': 0,
                'extra_distance': 0,
                'extra_duration': 0,
                'detour_ratio': 1.0
            } for _ in pois]
        
        # 組合結果
        recommendations = []
        for poi, score, detour in zip(pois, scores, detours):
            recommendations.append({
                'poi': poi,
                'score': float(score),
                'detour_info': detour,
                'extra_time_minutes': detour.get('extra_duration', 0) / 60.0 if detour.get('extra_duration') else 0,
                'reasons': self._generate_recommendation_reasons(
                    poi, user_profile or {}, score, detour
                )
            })
        
        # 按分數排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # 🎯 核心功能：LLM逐一審核
        if self.enable_llm_filter and self.llm_filter:
            print(f"\n🤖 開始LLM逐一審核流程...")
            print(f"   目標: TOP {top_k} 旅遊推薦")
            print(f"   候選: {len(recommendations)} 個排序結果")
            
            # 提取用戶偏好類別
            user_categories = []
            if user_history:
                user_categories = list(set([
                    item.get('category') 
                    for item in user_history 
                    if item.get('category')
                ]))
            
            if user_categories:
                print(f"   用戶偏好類別: {', '.join(user_categories)}")
            
            # 提取POI用於LLM審核
            ranked_pois = [rec['poi'] for rec in recommendations]
            
            # 根據配置選擇併發或順序模式
            if self.enable_llm_concurrent:
                # 使用併發版本（顯著提升速度）
                approved_pois = self.llm_filter.sequential_llm_filter_top_k_concurrent(
                    ranked_pois, 
                    target_k=top_k,
                    start_location=start_location,
                    end_location=end_location,
                    batch_size=10,  # 每批次併發10個
                    user_categories=user_categories if user_categories else None,
                    early_stop=True,
                    early_stop_buffer=1.5
                )
            else:
                # 使用順序版本（兼容模式）
                approved_pois = self.llm_filter.sequential_llm_filter_top_k(
                    ranked_pois, 
                    target_k=top_k,
                    start_location=start_location,
                    end_location=end_location,
                    multiplier=3,
                    user_categories=user_categories if user_categories else None,
                    early_stop=True,
                    early_stop_buffer=1.5
                )
            
            # 重新構建推薦結果（保持原始分數和詳細資訊）
            final_recommendations = []
            for approved_poi in approved_pois:
                # 找到對應的原始推薦資訊
                for rec in recommendations:
                    if rec['poi'] == approved_poi:
                        # 添加LLM審核標記
                        rec['llm_approved'] = True
                        final_recommendations.append(rec)
                        break
            
            print(f"\n✅ LLM審核完成!")
            print(f"   最終推薦: {len(final_recommendations)} 個")
            
            return final_recommendations
        else:
            # 不使用LLM過濾，直接返回top-k
            return recommendations[:top_k]
    
    def _update_performance_stats(self, total_time: float):
        """更新性能統計"""
        count = self.performance_stats['total_recommendations']
        self.performance_stats['avg_recommendation_time'] = (
            (self.performance_stats['avg_recommendation_time'] * (count - 1) + total_time) / count
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """獲取性能報告"""
        report = self.performance_stats.copy()
        
        if self.spatial_index:
            report['spatial_index_stats'] = self.spatial_index.get_index_stats()
        
        report['osrm_stats'] = self.osrm_client.get_performance_stats()
        
        return report
    
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
        
        # 準備路徑特徵（使用 POI 到直線的距離）
        path_continuous_list = []
        for poi in pois:
            poi_location = (poi['latitude'], poi['longitude'])
            
            # 計算 POI 到起點-終點直線的垂直距離
            perpendicular_dist = self._point_to_line_distance(
                poi_location, start_location, end_location
            )
            
            # 計算其他距離指標
            dist_to_start = self._haversine_distance(
                start_location[0], start_location[1],
                poi_location[0], poi_location[1]
            )
            dist_to_end = self._haversine_distance(
                poi_location[0], poi_location[1],
                end_location[0], end_location[1]
            )
            direct_dist = self._haversine_distance(
                start_location[0], start_location[1],
                end_location[0], end_location[1]
            )
            
            # 計算繞道比例
            total_dist = dist_to_start + dist_to_end
            detour_ratio = total_dist / direct_dist if direct_dist > 0 else 1.0
            
            path_features = np.array([
                min(perpendicular_dist / 50.0, 1.0),  # POI到直線距離（主要指標）
                min(dist_to_start / 100.0, 1.0),      # 到起點距離
                min(dist_to_end / 100.0, 1.0),        # 到終點距雩
                min((detour_ratio - 1.0), 1.0)        # 繞道比例
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
        detour: Dict
    ) -> List[str]:
        """生成個性化推薦理由"""
        reasons = []
        priority_reasons = []  # 高優先級理由
        
        # 1. 個人化匹配理由 (最重要)
        poi_category = poi.get('primary_category', '')
        user_categories = user_profile.get('preferred_categories', [])
        
        if poi_category in user_categories:
            category_rank = user_categories.index(poi_category) + 1
            if category_rank == 1:
                priority_reasons.append(f"完美符合您的最愛類型 ({poi_category})")
            elif category_rank <= 3:
                priority_reasons.append(f"符合您的偏好 ({poi_category}，排名第{category_rank})")
        
        # 2. 質量與可信度
        rating = poi.get('avg_rating', 0)
        num_reviews = poi.get('num_reviews', 0)
        
        if rating >= 4.8 and num_reviews >= 50:
            priority_reasons.append(f"頂級評分景點 ({rating:.1f}⭐，{num_reviews}+ 評論)")
        elif rating >= 4.5 and num_reviews >= 20:
            reasons.append(f"⭐ 高評分認證 ({rating:.1f}⭐，{num_reviews} 條評論)")
        elif rating >= 4.0:
            reasons.append(f"� 好評推薦 ({rating:.1f}⭐)")
        
        # 3. 路線便利性 (考慮用戶時間偏好)
        extra_minutes = detour.get('extra_duration', 0) / 60.0 if detour.get('extra_duration') else 0
        detour_ratio = detour.get('detour_ratio', 1.0)
        
        if extra_minutes > 0:
            if extra_minutes < 10:
                priority_reasons.append(f"幾乎順路 (僅需額外 {extra_minutes:.0f} 分鐘)")
            elif extra_minutes < 20:
                reasons.append(f"輕鬆到達 (額外 {extra_minutes:.0f} 分鐘)")
            elif extra_minutes < 30 and (detour_ratio - 1.0) < 0.3:
                reasons.append(f"適度繞行 (額外 {extra_minutes:.0f} 分鐘，值得一訪)")
        else:
            # 沒有 detour 信息時，基於類別和評分推薦
            if rating >= 4.5:
                reasons.append("地理位置優越")
        
        # 4. 價格與價值
        price_level = poi.get('price_level', 0)
        user_avg_price = user_profile.get('avg_price_level', 2.0)
        
        if price_level <= user_avg_price - 0.5:
            reasons.append("超值選擇")
        elif price_level <= user_avg_price:
            reasons.append("價格合理")
        elif price_level == 0:
            reasons.append("免費景點")
        
        # 5. 特色與便利性
        special_features = []
        if poi.get('is_open_24h', False):
            special_features.append("24小時營業")
        if poi.get('wheelchair_accessible', False):
            special_features.append("無障礙設施")
        if poi.get('good_for_groups', False):
            special_features.append("適合團體")
        if poi.get('has_parking', False):
            special_features.append("有停車場")
        if poi.get('pet_friendly', False):
            special_features.append("寵物友善")
        
        if special_features:
            reasons.append(f"✨ {special_features[0]}")
        
        # 6. 熱門度與趨勢
        if num_reviews > 500:
            reasons.append(f"超人氣景點 ({num_reviews}+ 遊客推薦)")
        elif num_reviews > 100:
            reasons.append(f"熱門選擇 ({num_reviews} 條評論)")
        
        # 7. 推薦強度
        if score > 0.85:
            priority_reasons.append("AI 強烈推薦!")
        elif score > 0.75:
            reasons.append("AI 推薦")
        
        # 8. 獨特性與發現價值
        if num_reviews < 20 and rating >= 4.2:
            reasons.append("隱藏寶石 (小眾但高品質)")
        
        # 9. 季節性或時間相關
        import datetime
        current_hour = datetime.datetime.now().hour
        if poi.get('good_for_evening', False) and current_hour >= 17:
            reasons.append("夜晚好去處")
        elif poi.get('good_for_morning', False) and current_hour <= 11:
            reasons.append("晨間推薦")
        
        # 組合最終理由 (優先級理由 + 一般理由)
        final_reasons = priority_reasons[:2] + reasons
        
        # 確保至少有一個理由
        if not final_reasons:
            final_reasons.append(f"推薦景點 (評分 {rating:.1f})")
        
        return final_reasons[:4]  # 最多返回4個理由，提供更豐富的資訊


def create_route_recommender(
    poi_data_path: str = "datasets/meta-California.json.gz",
    model_checkpoint: Optional[str] = None,
    device: str = 'cpu',
    enable_spatial_index: bool = True,
    enable_async: bool = True
) -> RouteAwareRecommender:
    """
    創建路徑感知推薦器 - 優化版
    
    Args:
        poi_data_path: POI數據路徑
        model_checkpoint: 模型檢查點路徑
        device: 運算設備
        enable_spatial_index: 啟用空間索引
        enable_async: 啟用異步處理
    
    Returns:
        RouteAwareRecommender 實例
    """
    print("正在初始化路徑感知推薦器...")
    
    # 載入POI數據
    try:
        from data_processor import POIDataProcessor
        poi_processor = POIDataProcessor(poi_data_path)
        poi_processor.load_data(max_records=1000000)
        poi_processor.preprocess()
        
        print(f"✓ POI數據載入成功")
        
    except Exception as e:
        print(f"❌ POI數據載入失敗: {e}")
        print(f"嘗試使用模擬數據...")
        
        # 創建模擬 POI 處理器
        class MockPOIProcessor:
            def __init__(self):
                self.pois = []  # 空列表
                self.poi_index = {}
                self.category_encoder = {}
                self.state_encoder = {}
                
            def encode_poi(self, poi):
                return {
                    'continuous': [0.5, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'categorical': {
                        'category': 0,
                        'state': 0,
                        'price_level': 2
                    }
                }
                
            def get_pois_by_location(self, lat, lon, radius_km):
                return []  # 返回空列表
        
        poi_processor = MockPOIProcessor()
    
    # 創建模型
    try:
        # 設置預設的詞彙表大小
        poi_vocab_sizes = {
            'category': getattr(poi_processor, 'category_encoder', {}) and len(poi_processor.category_encoder) or 100,
            'state': getattr(poi_processor, 'state_encoder', {}) and len(poi_processor.state_encoder) or 50,
            'price_level': 5
        }
        
        print(f"   模型詞彙表大小: {poi_vocab_sizes}")
        
        model = create_travel_dlrm(
            user_continuous_dim=10,
            poi_continuous_dim=8,
            path_continuous_dim=4,
            user_vocab_sizes={},
            poi_vocab_sizes=poi_vocab_sizes,
            embedding_dim=64
        )
        
        print(f"✓ 模型創建成功")
        
    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        # 創建模擬模型
        class MockModel:
            def to(self, device): return self
            def eval(self): return self
            def predict(self, *args): 
                import numpy as np
                return np.random.rand(args[0].shape[0] if hasattr(args[0], 'shape') else 1)
        model = MockModel()
    
    # 載入模型權重
    if model_checkpoint:
        try:
            print(f"載入模型權重: {model_checkpoint}")
            checkpoint = torch.load(model_checkpoint, map_location=device)
            
            # 檢查模型相容性
            if hasattr(model, 'load_state_dict'):
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ 模型權重載入成功")
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        print(f"⚠️ 模型結構不匹配: {e}")
                        print(f"使用預設模型參數")
                    else:
                        raise e
            else:
                print(f"⚠️ 模擬模型不支援權重載入")
                
        except Exception as e:
            print(f"❌ 模型權重載入失敗: {e}")
            print(f"使用預設模型參數")
    
    # 創建OSRM客戶端
    osrm_client = OSRMClient()
    
    # 創建推薦器
    try:
        recommender = RouteAwareRecommender(
            model=model,
            poi_processor=poi_processor,
            osrm_client=osrm_client,
            device=device,
            enable_spatial_index=enable_spatial_index,
            enable_async=enable_async
        )
        
        print(f"✅ 路徑感知推薦器初始化完成!")
        return recommender
        
    except Exception as e:
        print(f"❌ 推薦器初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        raise e


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
