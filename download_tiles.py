#!/usr/bin/env python3
"""
OSM 瓦片下載器 - 生成離線 MBTiles 檔案
用於 Leaflet 離線地圖

使用方式：
    python3 download_tiles.py --bbox 37.6,-122.6,37.9,-122.2 --zoom 8-16 --output sf_bay_area.mbtiles
"""

import os
import sys
import argparse
import sqlite3
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math

class TileDownloader:
    def __init__(self, output_file, max_workers=8):
        self.output_file = output_file
        self.max_workers = max_workers
        self.tile_server = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        self.user_agent = "RouteX-TileDownloader/1.0"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
        # 初始化資料庫
        self.init_database()
        
    def init_database(self):
        """初始化 MBTiles 資料庫結構"""
        conn = sqlite3.connect(self.output_file)
        cursor = conn.cursor()
        
        # 創建 metadata 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # 創建 tiles 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                zoom_level INTEGER,
                tile_column INTEGER,
                tile_row INTEGER,
                tile_data BLOB,
                PRIMARY KEY (zoom_level, tile_column, tile_row)
            )
        """)
        
        # 插入元數據
        metadata = {
            'name': 'RouteX San Francisco Bay Area',
            'type': 'baselayer',
            'version': '1.0',
            'description': 'Offline tiles for San Francisco Bay Area',
            'format': 'png',
            'attribution': '© OpenStreetMap contributors'
        }
        
        for key, value in metadata.items():
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)",
                (key, value)
            )
        
        conn.commit()
        conn.close()
        print(f" 資料庫初始化完成: {self.output_file}")
    
    def latlon_to_tile(self, lat, lon, zoom):
        """將經緯度轉換為瓦片座標"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
    
    def get_tile_bounds(self, min_lat, min_lon, max_lat, max_lon, zoom):
        """獲取指定範圍的所有瓦片座標"""
        x1, y1 = self.latlon_to_tile(max_lat, min_lon, zoom)
        x2, y2 = self.latlon_to_tile(min_lat, max_lon, zoom)
        
        # 確保正確的順序
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((zoom, x, y))
        
        return tiles
    
    def download_tile(self, zoom, x, y):
        """下載單個瓦片"""
        url = self.tile_server.format(z=zoom, x=x, y=y)
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # MBTiles 使用 TMS 座標系統，需要翻轉 Y 軸
            tms_y = (2 ** zoom) - 1 - y
            
            return (zoom, x, tms_y, response.content)
        except Exception as e:
            print(f" 下載失敗 [{zoom}/{x}/{y}]: {e}")
            return None
    
    def save_tile(self, zoom, x, y, data):
        """保存瓦片到資料庫"""
        conn = sqlite3.connect(self.output_file)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
                (zoom, x, y, data)
            )
            conn.commit()
        except Exception as e:
            print(f" 保存失敗 [{zoom}/{x}/{y}]: {e}")
        finally:
            conn.close()
    
    def download_tiles(self, min_lat, min_lon, max_lat, max_lon, zoom_min, zoom_max):
        """下載所有瓦片"""
        total_tiles = 0
        downloaded_tiles = 0
        
        # 計算總瓦片數
        for zoom in range(zoom_min, zoom_max + 1):
            tiles = self.get_tile_bounds(min_lat, min_lon, max_lat, max_lon, zoom)
            total_tiles += len(tiles)
        
        print(f"\n 開始下載瓦片")
        print(f"   範圍: [{min_lat}, {min_lon}] 到 [{max_lat}, {max_lon}]")
        print(f"   縮放級別: {zoom_min} - {zoom_max}")
        print(f"   總瓦片數: {total_tiles}")
        print(f"   並發數: {self.max_workers}")
        print("=" * 60)
        
        # 下載瓦片
        for zoom in range(zoom_min, zoom_max + 1):
            tiles = self.get_tile_bounds(min_lat, min_lon, max_lat, max_lon, zoom)
            print(f"\n 縮放級別 {zoom}: {len(tiles)} 個瓦片")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_tile, zoom, x, y): (zoom, x, y)
                    for zoom, x, y in tiles
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        zoom, x, y, data = result
                        self.save_tile(zoom, x, y, data)
                        downloaded_tiles += 1
                        
                        if downloaded_tiles % 100 == 0:
                            progress = (downloaded_tiles / total_tiles) * 100
                            print(f"   進度: {downloaded_tiles}/{total_tiles} ({progress:.1f}%)")
                    
                    # 延遲以避免服務器限制
                    time.sleep(0.1)
        
        print(f"\n 下載完成！")
        print(f"   成功: {downloaded_tiles}/{total_tiles}")
        print(f"   輸出: {self.output_file}")
        
        # 顯示文件大小
        file_size = os.path.getsize(self.output_file) / (1024 * 1024)
        print(f"   大小: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='下載 OSM 瓦片並生成 MBTiles 檔案')
    parser.add_argument('--bbox', type=str, required=True,
                        help='邊界框 (min_lat,min_lon,max_lat,max_lon)，例如: 37.6,-122.6,37.9,-122.2')
    parser.add_argument('--zoom', type=str, default='8-16',
                        help='縮放級別範圍 (min-max)，例如: 8-16')
    parser.add_argument('--output', type=str, default='tiles.mbtiles',
                        help='輸出 MBTiles 檔案名稱')
    parser.add_argument('--workers', type=int, default=8,
                        help='並發下載數 (預設: 8)')
    
    args = parser.parse_args()
    
    # 解析邊界框
    try:
        min_lat, min_lon, max_lat, max_lon = map(float, args.bbox.split(','))
    except:
        print(" 邊界框格式錯誤！應為: min_lat,min_lon,max_lat,max_lon")
        sys.exit(1)
    
    # 解析縮放級別
    try:
        zoom_min, zoom_max = map(int, args.zoom.split('-'))
    except:
        print(" 縮放級別格式錯誤！應為: min-max")
        sys.exit(1)
    
    # 驗證參數
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        print(" 緯度必須在 -90 到 90 之間")
        sys.exit(1)
    
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        print(" 經度必須在 -180 到 180 之間")
        sys.exit(1)
    
    if not (0 <= zoom_min <= 18 and 0 <= zoom_max <= 18 and zoom_min <= zoom_max):
        print(" 縮放級別必須在 0 到 18 之間，且 min <= max")
        sys.exit(1)
    
    # 創建輸出目錄
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("️  OSM 瓦片下載器")
    print("=" * 60)
    
    # 下載瓦片
    downloader = TileDownloader(args.output, max_workers=args.workers)
    downloader.download_tiles(min_lat, min_lon, max_lat, max_lon, zoom_min, zoom_max)
    
    print("\n" + "=" * 60)
    print(" 完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
