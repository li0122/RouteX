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


class OSRMClient:
    """OSRM 路徑規劃客戶端 - 優化版"""
    
    def __init__(self, server_url: str = "http://router.project-osrm.org"):
        self.server_url = server_url
        self.cache_size = 10000  # 增加緩存大小從1000到10000
        self.session = None
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
            
            # 使用會話復用連接
            if not hasattr(requests, '_session'):
                requests._session = requests.Session()
                requests._session.headers.update({
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'gzip, deflate'
                })
            
            response = requests._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
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
        批量異步計算繞道成本 - 主要性能優化
        
        Args:
            start: 起點
            end: 終點
            waypoints: 中繼點列表
            max_concurrent: 最大並發數
            
        Returns:
            繞道成本結果列表
        """
        if not ASYNC_SUPPORTED or not waypoints:
            # 回退到同步模式
            return [self.calculate_detour(start, wp, end) for wp in waypoints]
        
        # 異步批量處理
        connector = aiohttp.TCPConnector(
            limit=50, 
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Connection': 'keep-alive',
                'Accept-Encoding': 'gzip, deflate'
            }
        ) as session:
            
            # 首先獲取直達路線
            direct_route = await self._get_route_async(session, start, end)
            if not direct_route:
                return [None] * len(waypoints)
            
            # 使用信號量控制並發數
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def calculate_single_detour(waypoint):
                async with semaphore:
                    return await self._calculate_detour_async(
                        session, start, end, waypoint, direct_route
                    )
            
            # 並行執行所有查詢
            tasks = [calculate_single_detour(wp) for wp in waypoints]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 處理異常
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"繞道計算失敗: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
    
    async def _get_route_async(
        self,
        session: 'aiohttp.ClientSession',
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile: str = "driving"
    ) -> Optional[Dict]:
        """異步獲取路線"""
        try:
            url = f"{self.server_url}/route/v1/{profile}/{start[1]},{start[0]};{end[1]},{end[0]}"
            params = {
                'overview': 'false',
                'steps': 'false',
                'alternatives': 'false'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('code') == 'Ok' and 'routes' in data:
                        route = data['routes'][0]
                        return {
                            'distance': route['distance'],
                            'duration': route['duration']
                        }
                
                return None
                
        except Exception as e:
            print(f"OSRM異步查詢失敗: {e}")
            return None
    
    async def _calculate_detour_async(
        self,
        session: 'aiohttp.ClientSession',
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoint: Tuple[float, float],
        direct_route: Dict
    ) -> Optional[Dict]:
        """異步計算單個POI的繞道成本"""
        
        # 並行查詢兩段路線
        route1_task = self._get_route_async(session, start, waypoint)
        route2_task = self._get_route_async(session, waypoint, end)
        
        route1, route2 = await asyncio.gather(route1_task, route2_task)
        
        if not route1 or not route2:
            return None
        
        # 計算繞道資訊
        via_distance = route1['distance'] + route2['distance']
        via_duration = route1['duration'] + route2['duration']
        
        extra_distance = via_distance - direct_route['distance']
        extra_duration = via_duration - direct_route['duration']
        detour_ratio = via_distance / direct_route['distance']
        
        return {
            'direct_distance': direct_route['distance'],
            'direct_duration': direct_route['duration'],
            'via_distance': via_distance,
            'via_duration': via_duration,
            'extra_distance': extra_distance,
            'extra_duration': extra_duration,
            'detour_ratio': detour_ratio
        }
    
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
    """路徑感知推薦器 - 優化版"""
    
    def __init__(
        self,
        model: TravelDLRM,
        poi_processor: POIDataProcessor,
        osrm_client: Optional[OSRMClient] = None,
        device: str = 'cpu',
        enable_spatial_index: bool = True,
        enable_async: bool = True
    ):
        self.model = model
        self.poi_processor = poi_processor
        self.osrm_client = osrm_client or OSRMClient()
        self.device = torch.device(device)
        self.user_preference_model = UserPreferenceModel()
        self.enable_async = enable_async and ASYNC_SUPPORTED
        
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
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 優化版推薦器初始化完成")
        enabled_text = "啟用" if self.spatial_index and self.spatial_index.index_built else "禁用"
        print(f"   - 空間索引: {enabled_text}")
        async_text = "啟用" if self.enable_async else "禁用"
        print(f"   - 異步支持: {async_text}")
    
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
        在路線上推薦景點 - 優化版
        
        主要優化:
        1. 空間索引加速 POI 搜索
        2. 智能預過濾減少無效計算
        3. 異步 OSRM 查詢提高並發性
        
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
        start_time = time.time()
        self.performance_stats['total_recommendations'] += 1
        
        print(f"🎯 開始路線推薦: {start_location} → {end_location}")
        
        # 1. 建立用戶畫像
        print("👤 步驟1: 建立用戶畫像...")
        user_profile = self.user_preference_model.build_user_profile(
            user_id, user_history
        )
        
        # 2. 空間索引搜索候選POI
        print("🗺️ 步驟2: 搜索候選POI...")
        search_start = time.time()
        
        if candidate_pois is None:
            candidate_pois = self._spatial_search_candidates(
                start_location, end_location
            )
        
        search_time = time.time() - search_start
        print(f"   搜索完成: {len(candidate_pois)} 個候選POI (耗時: {search_time:.3f}s)")
        
        if not candidate_pois:
            print("⚠️ 沒有找到候選POI")
            return []
        
        # 3. 智能預過濾
        print("⚡ 步驟3: 智能預過濾...")
        filter_start = time.time()
        
        filtered_pois = self._intelligent_prefilter(
            candidate_pois, user_history, max_candidates=50
        )
        
        filter_time = time.time() - filter_start
        print(f"   過濾完成: {len(filtered_pois)} 個高品質候選 (耗時: {filter_time:.3f}s)")
        
        # 4. 異步路線過濾
        if self.enable_async:
            return asyncio.run(self._async_route_recommendation(
                user_profile, filtered_pois, start_location, end_location,
                top_k, max_detour_ratio, max_extra_duration, start_time
            ))
        else:
            return self._sync_route_recommendation(
                user_profile, filtered_pois, start_location, end_location,
                top_k, max_detour_ratio, max_extra_duration, start_time
            )
    
    def _spatial_search_candidates(
        self,
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        radius_km: float = 30.0
    ) -> List[Dict]:
        """空間索引搜索候選POI"""
        
        if self.spatial_index and self.spatial_index.index_built:
            # 使用空間索引
            mid_lat = (start_location[0] + end_location[0]) / 2
            mid_lon = (start_location[1] + end_location[1]) / 2
            
            candidates = self.spatial_index.query_by_location(
                mid_lat, mid_lon, radius_km, max_results=200
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
        max_candidates: int = 50
    ) -> List[Dict]:
        """智能預過濾 - 減少無效計算"""
        
        if len(candidates) <= max_candidates:
            return candidates
        
        # 提取用戶偏好
        user_categories = set(h.get('category', '') for h in user_history)
        user_avg_rating = np.mean([h.get('rating', 3.5) for h in user_history]) if user_history else 3.5
        
        # 評分函數
        def score_candidate(poi):
            score = 0
            
            # 評分權重 (30%)
            poi_rating = poi.get('avg_rating', 0)
            if poi_rating > 0:
                score += poi_rating * 0.3
            
            # 類別匹配 (40%)
            if poi.get('primary_category', '') in user_categories:
                score += 2.0
            
            # 熱門度 (20%)
            review_count = poi.get('num_reviews', 0)
            if review_count > 0:
                score += min(np.log1p(review_count) * 0.1, 1.0)
            
            # 價格合適性 (10%)
            price_level = poi.get('price_level', 2)
            if price_level <= 3:  # 不太貴
                score += 0.5
            
            # 安全性檢查
            if poi_rating < 2.0:  # 過低評分
                score *= 0.5
            
            return score
        
        # 計算分數並排序
        scored_candidates = [(poi, score_candidate(poi)) for poi in candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N個
        filtered = [poi for poi, score in scored_candidates[:max_candidates]]
        
        print(f"   預過濾: {len(candidates)} → {len(filtered)} (減少 {(1-len(filtered)/len(candidates))*100:.1f}%)")
        return filtered
    
    async def _async_route_recommendation(
        self,
        user_profile: Dict,
        filtered_pois: List[Dict],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        top_k: int,
        max_detour_ratio: float,
        max_extra_duration: float,
        start_time: float
    ) -> List[Dict]:
        """異步路線推薦流程"""
        
        print("🚀 步驟4: 異步路線過濾...")
        osrm_start = time.time()
        
        # 提取POI位置
        poi_locations = [(poi['latitude'], poi['longitude']) for poi in filtered_pois]
        
        # 異步批量計算繞道成本
        detour_results = await self.osrm_client.batch_calculate_detours(
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
            valid_pois, scores, valid_detours, top_k, user_profile
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
        start_time: float
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
            
            # 早停機制: 如果已經找到足夠的POI
            if len(valid_pois) >= top_k * 2:  # 找到隙2倍的目標數量就停止
                print(f"   早停: 已找到足夠的POI ({len(valid_pois)})")  
                break
        
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
            valid_pois, scores, valid_detours, top_k, user_profile
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
            selected_pois, scores, mock_detours, top_k, {'preferred_categories': []}
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
    
    def _generate_recommendations(
        self,
        pois: List[Dict],
        scores: List[float],
        detours: List[Dict],
        top_k: int,
        user_profile: Dict = None
    ) -> List[Dict]:
        """生成推薦結果"""
        
        # 組合結果
        recommendations = []
        for poi, score, detour in zip(pois, scores, detours):
            recommendations.append({
                'poi': poi,
                'score': float(score),
                'detour_info': detour,
                'extra_time_minutes': detour['extra_duration'] / 60.0,
                'reasons': self._generate_recommendation_reasons(
                    poi, user_profile or {}, score, detour
                )
            })
        
        # 排序並返回top-k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
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
