"""
數據處理模組
處理 datasets/ 中的旅遊與景點數據
支援 .json 和 .json.gz 格式
"""

import json
import gzip
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import pickle
from pathlib import Path


class POIDataProcessor:
    """POI (景點) 數據處理器"""
    
    def __init__(self, dataset_path: str = "datasets/meta-California.json.gz"):
        """初始化 POI 數據處理器
        
        Args:
            dataset_path: 資料集路徑，支援 .json 或 .json.gz
                         預設使用 travel_recommender/datasets/ 下的 California 資料集
        """
        self.dataset_path = dataset_path
        self.pois = []
        self.raw_pois = []
        self.processed_pois = []
        self.poi_index = {}
        
        # 類別編碼
        self.category_encoder = {}
        self.category_decoder = {}
        self.category_counter = Counter()
        
        # 城市編碼
        self.state_encoder = {}
        self.state_decoder = {}
        
        # 統計信息
        self.stats = {}
    
    def load_data(self, max_records: Optional[int] = None) -> List[Dict]:
        """
        載入POI數據（支援 .json 和 .json.gz）
        
        Args:
            max_records: 最大載入記錄數 (None表示全部載入)
        
        Returns:
            POI數據列表
        """
        print(f"正在載入 POI 數據: {self.dataset_path}")
        
        pois = []
        
        # 判斷是否為壓縮檔案
        is_gzip = self.dataset_path.endswith('.gz')
        open_func = gzip.open if is_gzip else open
        mode = 'rt' if is_gzip else 'r'
        
        try:
            with open_func(self.dataset_path, mode, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_records and i >= max_records:
                        break
                    
                    if i > 0 and i % 10000 == 0:
                        print(f"  已載入 {i} 筆記錄...")
                    
                    try:
                        poi = json.loads(line.strip())
                        pois.append(poi)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"❌ 找不到檔案: {self.dataset_path}")
            print(f"請確認檔案位於正確位置")
            return []
        
        self.raw_pois = pois
        self.pois = pois

        print(f"✓ 成功載入 {len(pois)} 個 POI")
        
        return pois
    
    def preprocess(self) -> Dict[str, Any]:
        """
        預處理POI數據
        
        Returns:
            處理後的數據字典
        """
        print("開始預處理 POI 數據...")
        
        processed_pois = []
        
        for idx, poi in enumerate(self.raw_pois):
            # 提取基本信息
            gmap_id = poi.get('gmap_id', f'poi_{idx}')
            name = poi.get('name', 'Unknown')
            
            # 地理坐標
            latitude = poi.get('latitude', 0.0)
            longitude = poi.get('longitude', 0.0)
            
            # 評分信息
            avg_rating = poi.get('avg_rating', 0.0)
            num_reviews = poi.get('num_of_reviews', 0)
            
            # 類別信息
            categories = poi.get('category', [])
            if isinstance(categories, list) and categories:
                primary_category = categories[0]
                self.category_counter[primary_category] += 1
            else:
                primary_category = 'Other'
            
            # 價格水平
            price = poi.get('price', None)
            if price:
                price_level = len(price)  # $ -> 1, $$ -> 2, etc.
            else:
                price_level = 0
            
            # 狀態/城市
            state = poi.get('state', 'Unknown')
            
            # 營業時間
            hours = poi.get('hours', None)
            is_open_24h = False
            if hours and isinstance(hours, list):
                for h in hours:
                    if isinstance(h, list) and len(h) > 1:
                        if 'Open 24 hours' in h[1] or '24' in h[1]:
                            is_open_24h = True
                            break
            
            processed_poi = {
                'id': gmap_id,
                'name': name,
                'latitude': float(latitude) if latitude else 0.0,
                'longitude': float(longitude) if longitude else 0.0,
                'avg_rating': float(avg_rating) if avg_rating else 0.0,
                'num_reviews': int(num_reviews) if num_reviews else 0,
                'categories': categories if categories else [],
                'primary_category': primary_category,
                'price_level': price_level,
                'state': state,
                'is_open_24h': is_open_24h,
                'url': poi.get('url', ''),
                'address': poi.get('address', ''),
                'description': poi.get('description', '')
            }
            
            processed_pois.append(processed_poi)
            self.poi_index[gmap_id] = idx
        
        # 建立編碼器
        self._build_encoders()
        
        # 計算統計信息
        self._compute_statistics(processed_pois)
        
        print(f"預處理完成! 共 {len(processed_pois)} 個有效 POI")
        self.processed_pois = processed_pois
        self.pois = processed_pois
        
        return {
            'pois': processed_pois,
            'category_encoder': self.category_encoder,
            'state_encoder': self.state_encoder,
            'stats': self.stats
        }
    
    def _build_encoders(self):
        """建立類別和狀態的編碼器"""
        # 類別編碼 (只保留前N個最常見的類別)
        top_categories = [cat for cat, _ in self.category_counter.most_common(100)]
        self.category_encoder = {cat: idx for idx, cat in enumerate(top_categories)}
        self.category_encoder['Other'] = len(top_categories)  # 未知類別
        self.category_decoder = {idx: cat for cat, idx in self.category_encoder.items()}
        
        # 狀態編碼
        states = set(poi.get('state', 'Unknown') for poi in self.pois if poi.get('state'))
        self.state_encoder = {state: idx for idx, state in enumerate(sorted(states))}
        self.state_encoder['Unknown'] = len(states)
        self.state_decoder = {idx: state for state, idx in self.state_encoder.items()}
    
    def _compute_statistics(self, processed_pois: List[Dict]):
        """計算統計信息"""
        ratings = [poi['avg_rating'] for poi in processed_pois if poi['avg_rating'] > 0]
        num_reviews = [poi['num_reviews'] for poi in processed_pois]
        
        self.stats = {
            'total_pois': len(processed_pois),
            'avg_rating_mean': np.mean(ratings) if ratings else 0,
            'avg_rating_std': np.std(ratings) if ratings else 0,
            'num_reviews_mean': np.mean(num_reviews) if num_reviews else 0,
            'num_reviews_std': np.std(num_reviews) if num_reviews else 0,
            'num_categories': len(self.category_encoder),
            'num_states': len(self.state_encoder),
            'top_categories': self.category_counter.most_common(10)
        }
    
    def encode_poi(self, poi: Dict) -> Dict[str, Any]:
        """
        將POI編碼為模型輸入格式
        
        Returns:
            {
                'continuous': numpy array,
                'categorical': dict of indices
            }
        """
        # 連續特徵
        continuous_features = np.array([
            poi.get('avg_rating', 0.0),
            np.log1p(poi.get('num_reviews', 0)),  # log變換
            poi.get('price_level', 0) / 4.0,  # 標準化到 [0, 1]
            poi.get('latitude', 0.0) / 90.0,  # 標準化緯度
            poi.get('longitude', 0.0) / 180.0,  # 標準化經度
            float(poi.get('is_open_24h', False)),
            # 可以添加更多特徵
            0.0,  # 預留
            0.0   # 預留
        ], dtype=np.float32)
        
        # 類別特徵
        primary_cat = poi.get('primary_category', 'Other')
        category_idx = self.category_encoder.get(primary_cat, self.category_encoder['Other'])
        
        state = poi.get('state', 'Unknown')
        state_idx = self.state_encoder.get(state, self.state_encoder['Unknown'])
        
        categorical_features = {
            'category': category_idx,
            'state': state_idx,
            'price_level': poi.get('price_level', 0)
        }
        
        return {
            'continuous': continuous_features,
            'categorical': categorical_features,
            'raw': poi
        }
    
    def get_pois_by_location(
        self, 
        center_lat: float, 
        center_lon: float, 
        radius_km: float = 10.0
    ) -> List[Dict]:
        """
        根據地理位置獲取附近的POI
        
        Args:
            center_lat: 中心緯度
            center_lon: 中心經度
            radius_km: 搜索半徑 (公里)
        
        Returns:
            附近的POI列表
        """
        nearby_pois = []
        
        for poi in self.pois:
            lat = poi.get('latitude', 0.0)
            lon = poi.get('longitude', 0.0)
            
            if lat == 0.0 and lon == 0.0:
                continue
            
            # 計算距離 (簡化的haversine公式)
            distance = self._haversine_distance(center_lat, center_lon, lat, lon)
            
            if distance <= radius_km:
                nearby_pois.append(poi)
        
        return nearby_pois
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """計算兩點間的距離 (公里)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # 地球半徑 (公里)
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def save(self, filepath: str):
        """保存處理器狀態"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'poi_index': self.poi_index,
                'category_encoder': self.category_encoder,
                'category_decoder': self.category_decoder,
                'state_encoder': self.state_encoder,
                'state_decoder': self.state_decoder,
                'stats': self.stats
            }, f)
        print(f"數據處理器已保存到 {filepath}")
    
    def load(self, filepath: str):
        """載入處理器狀態"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.poi_index = data['poi_index']
            self.category_encoder = data['category_encoder']
            self.category_decoder = data['category_decoder']
            self.state_encoder = data['state_encoder']
            self.state_decoder = data['state_decoder']
            self.stats = data['stats']
        print(f"數據處理器已從 {filepath} 載入")


class ReviewDataProcessor:
    """評論數據處理器"""
    
    def __init__(self, dataset_path: str = "datasets/review-California.json.gz"):
        """初始化評論數據處理器
        
        Args:
            dataset_path: 資料集路徑，支援 .json 或 .json.gz
                         預設使用 travel_recommender/datasets/ 下的 California 資料集
        """
        self.dataset_path = dataset_path
        self.reviews = []
        self.user_reviews = defaultdict(list)
        self.poi_reviews = defaultdict(list)
    
    def load_data(self, max_records: Optional[int] = None) -> List[Dict]:
        """載入評論數據（支援 .json 和 .json.gz）"""
        print(f"正在載入評論數據: {self.dataset_path}")
        
        reviews = []
        
        # 判斷是否為壓縮檔案
        is_gzip = self.dataset_path.endswith('.gz')
        open_func = gzip.open if is_gzip else open
        mode = 'rt' if is_gzip else 'r'
        
        try:
            with open_func(self.dataset_path, mode, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_records and i >= max_records:
                        break
                    
                    if i > 0 and i % 50000 == 0:
                        print(f"  已載入 {i} 筆評論...")
                    
                    try:
                        review = json.loads(line.strip())
                        reviews.append(review)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"❌ 找不到檔案: {self.dataset_path}")
            print(f"請確認檔案位於正確位置")
            return []
        
        self.reviews = reviews
        print(f"✓ 成功載入 {len(reviews)} 條評論")
        
        return reviews
    
    def preprocess(self) -> Dict[str, Any]:
        """預處理評論數據"""
        print("開始預處理評論數據...")
        
        for review in self.reviews:
            user_id = review.get('user_id')
            gmap_id = review.get('gmap_id')
            
            if user_id and gmap_id:
                self.user_reviews[user_id].append(review)
                self.poi_reviews[gmap_id].append(review)
        
        print(f"預處理完成!")
        print(f"  用戶數: {len(self.user_reviews)}")
        print(f"  POI數: {len(self.poi_reviews)}")
        
        return {
            'user_reviews': dict(self.user_reviews),
            'poi_reviews': dict(self.poi_reviews)
        }
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        獲取用戶畫像
        
        Returns:
            {
                'avg_rating': float,
                'num_reviews': int,
                'visited_categories': List[str],
                'review_texts': List[str]
            }
        """
        user_data = self.user_reviews.get(user_id, [])
        
        if not user_data:
            return {
                'avg_rating': 0.0,
                'num_reviews': 0,
                'visited_categories': [],
                'review_texts': []
            }
        
        ratings = [r.get('rating', 0) for r in user_data if r.get('rating')]
        texts = [r.get('text', '') for r in user_data if r.get('text')]
        
        return {
            'avg_rating': np.mean(ratings) if ratings else 0.0,
            'num_reviews': len(user_data),
            'visited_categories': [],  # 需要與POI數據join
            'review_texts': texts
        }
    
    def create_user_poi_pairs(
        self, 
        min_rating: float = 4.0,
        negative_ratio: int = 2
    ) -> List[Tuple[str, str, int]]:
        """
        創建用戶-POI配對 (用於訓練)
        
        Args:
            min_rating: 正樣本最小評分
            negative_ratio: 負樣本比例
        
        Returns:
            (user_id, poi_id, label) 列表
        """
        pairs = []
        
        # 正樣本
        for user_id, reviews in self.user_reviews.items():
            for review in reviews:
                rating = review.get('rating', 0)
                if rating >= min_rating:
                    poi_id = review.get('gmap_id')
                    if poi_id:
                        pairs.append((user_id, poi_id, 1))
        
        print(f"生成 {len(pairs)} 個正樣本")
        
        # TODO: 生成負樣本 (隨機採樣未互動的POI)
        
        return pairs


def load_and_process_data(
    meta_path: str = "datasets/meta-California.json.gz",
    review_path: str = "datasets/review-California.json.gz",
    max_pois: Optional[int] = 10000,
    max_reviews: Optional[int] = 50000
) -> Tuple[POIDataProcessor, ReviewDataProcessor]:
    """
    載入並處理所有數據（支援壓縮檔案）
    
    Args:
        meta_path: POI 資料集路徑（支援 .json/.json.gz）
        review_path: 評論資料集路徑（支援 .json/.json.gz）
        max_pois: 最大載入 POI 數量
        max_reviews: 最大載入評論數量
    
    Returns:
        (POI處理器, 評論處理器)
    """
    print("\n" + "="*60)
    print("開始載入資料集...")
    print("="*60)
    # 處理POI數據
    poi_processor = POIDataProcessor(meta_path)
    poi_processor.load_data(max_records=max_pois)
    poi_processor.preprocess()
    
    # 處理評論數據
    review_processor = ReviewDataProcessor(review_path)
    review_processor.load_data(max_records=max_reviews)
    review_processor.preprocess()
    
    return poi_processor, review_processor


if __name__ == "__main__":
    print("=== 數據處理測試 ===\n")
    
    # 測試POI處理（使用較小的 other 資料集進行快速測試）
    print("使用測試資料集: datasets/meta-other.json")
    print("如需使用完整 California 資料集，請修改路徑為: datasets/meta-California.json.gz\n")
    
    poi_processor = POIDataProcessor("datasets/meta-California.json.gz")
    pois = poi_processor.load_data(max_records=1000000)
    
    result = poi_processor.preprocess()
    
    print(f"\n統計信息:")
    for key, value in poi_processor.stats.items():
        print(f"  {key}: {value}")
    
    # 測試編碼
    if result['pois']:
        sample_poi = result['pois'][0]
        print(f"\n範例 POI:")
        print(f"  名稱: {sample_poi['name']}")
        print(f"  類別: {sample_poi['primary_category']}")
        print(f"  評分: {sample_poi['avg_rating']}")
        
        encoded = poi_processor.encode_poi(sample_poi)
        print(f"\n編碼後:")
        print(f"  連續特徵: {encoded['continuous']}")
        print(f"  類別特徵: {encoded['categorical']}")
    
    # 測試地理搜索
    nearby = poi_processor.get_pois_by_location(37.7749, -122.4194, radius_km=10.0)
    print(f"\n舊金山市中心附近 10km 內有 {len(nearby)} 個 POI")
    
    print("\n✓ 數據處理測試完成!")
