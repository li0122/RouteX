#!/usr/bin/env python3
"""
OSRM 路線獲取工具
提供路線查詢、快取和錯誤處理
"""

import requests
import json
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class OSRMRouteClient:
    """OSRM 路線客戶端"""
    
    def __init__(self, 
                 server_url: str = "http://router.project-osrm.org",
                 cache_dir: Optional[str] = None,
                 timeout: int = 15):
        """
        初始化 OSRM 客戶端
        
        Args:
            server_url: OSRM 伺服器 URL
            cache_dir: 快取目錄路徑（None 則不使用快取）
            timeout: 請求超時時間（秒）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
        # 設置快取
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _generate_cache_key(self, waypoints: List[Tuple[float, float]], 
                           options: Dict) -> str:
        """生成快取鍵"""
        data = {
            'waypoints': waypoints,
            'options': options
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """從快取載入"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 檢查快取是否過期（24 小時）
                if time.time() - cached_data.get('timestamp', 0) < 86400:
                    return cached_data['data']
            except Exception as e:
                print(f"⚠️ 快取讀取失敗: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """保存到快取"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
        except Exception as e:
            print(f"⚠️ 快取寫入失敗: {e}")
    
    def get_route(self, 
                  waypoints: List[Tuple[float, float]],
                  geometries: str = 'geojson',
                  overview: str = 'full',
                  alternatives: bool = False,
                  use_cache: bool = True) -> Dict:
        """
        獲取路線
        
        Args:
            waypoints: 路徑點列表 [(lat1, lng1), (lat2, lng2), ...]
            geometries: 幾何格式 ('geojson', 'polyline', 'polyline6')
            overview: 概覽詳細程度 ('full', 'simplified', 'false')
            alternatives: 是否返回替代路線
            use_cache: 是否使用快取
        
        Returns:
            路線數據字典
        
        Raises:
            ValueError: 參數錯誤
            requests.RequestException: 請求失敗
        """
        # 驗證參數
        if not waypoints or len(waypoints) < 2:
            raise ValueError("至少需要 2 個路徑點")
        
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, (list, tuple)) or len(wp) != 2:
                raise ValueError(f"路徑點 {i} 格式錯誤，應為 (lat, lng)")
        
        # 檢查快取
        options = {
            'geometries': geometries,
            'overview': overview,
            'alternatives': alternatives
        }
        
        if use_cache:
            cache_key = self._generate_cache_key(waypoints, options)
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                print(f"✅ 從快取載入路線")
                return cached_data
        
        # 構建請求 URL
        coords = ';'.join([f"{lng},{lat}" for lat, lng in waypoints])
        url = f"{self.server_url}/route/v1/driving/{coords}"
        
        params = {
            'overview': overview,
            'geometries': geometries
        }
        
        if alternatives:
            params['alternatives'] = 'true'
        
        print(f"🗺️ 請求 OSRM 路線: {len(waypoints)} 個點")
        
        # 發送請求
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') != 'Ok':
                error_msg = data.get('message', 'Unknown error')
                raise ValueError(f"OSRM 錯誤: {data.get('code')} - {error_msg}")
            
            # 提取路線資訊
            route = data.get('routes', [{}])[0]
            
            result = {
                'code': 'Ok',
                'route': {
                    'geometry': route.get('geometry'),
                    'distance': route.get('distance'),  # 米
                    'duration': route.get('duration'),  # 秒
                    'legs': route.get('legs', [])
                },
                'waypoints': data.get('waypoints', [])
            }
            
            print(f"✅ OSRM 路線成功: {route.get('distance', 0)/1000:.1f} km, "
                  f"{route.get('duration', 0)/60:.0f} 分鐘")
            
            # 保存到快取
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
            
        except requests.Timeout:
            raise requests.RequestException("OSRM 請求超時")
        except requests.RequestException as e:
            raise requests.RequestException(f"OSRM 請求失敗: {e}")


def main():
    """命令行測試"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OSRM 路線查詢工具')
    parser.add_argument('--waypoints', type=str, required=True,
                        help='路徑點，格式: "lat1,lng1;lat2,lng2;..."')
    parser.add_argument('--server', type=str, default='http://router.project-osrm.org',
                        help='OSRM 伺服器 URL')
    parser.add_argument('--cache', type=str, default=None,
                        help='快取目錄')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用快取')
    
    args = parser.parse_args()
    
    # 解析路徑點
    try:
        waypoints = []
        for wp in args.waypoints.split(';'):
            lat, lng = map(float, wp.split(','))
            waypoints.append((lat, lng))
    except Exception as e:
        print(f"❌ 路徑點格式錯誤: {e}")
        print("   格式: 'lat1,lng1;lat2,lng2;...'")
        return 1
    
    # 創建客戶端
    client = OSRMRouteClient(
        server_url=args.server,
        cache_dir=args.cache
    )
    
    try:
        # 獲取路線
        result = client.get_route(
            waypoints=waypoints,
            use_cache=not args.no_cache
        )
        
        # 顯示結果
        route = result['route']
        print("\n" + "=" * 60)
        print("📍 路線資訊")
        print("=" * 60)
        print(f"距離: {route['distance'] / 1000:.2f} km")
        print(f"時間: {route['duration'] / 60:.0f} 分鐘")
        print(f"路徑點: {len(waypoints)}")
        
        if route.get('geometry'):
            geom = route['geometry']
            if isinstance(geom, dict) and 'coordinates' in geom:
                print(f"座標點數: {len(geom['coordinates'])}")
        
        # 輸出 JSON
        print("\n" + "=" * 60)
        print("JSON 輸出")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
