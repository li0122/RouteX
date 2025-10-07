#!/usr/bin/env python3
"""
資料集驗證腳本
用於檢查資料集是否正確放置和可讀取
"""

import json
import gzip
from pathlib import Path
from typing import Optional

def check_file(filepath: str) -> Optional[dict]:
    """檢查檔案是否存在且可讀取"""
    path = Path(filepath)
    
    if not path.exists():
        return None
    
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    
    # 嘗試讀取第一行以驗證格式
    is_gzip = filepath.endswith('.gz')
    open_func = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'
    
    try:
        with open_func(filepath, mode, encoding='utf-8') as f:
            first_line = f.readline()
            data = json.loads(first_line.strip())
            
            # 計算行數（僅前 1000 行以加快速度）
            count = 1
            for i, _ in enumerate(f):
                count += 1
                if i >= 999:  # 已讀 1 行 + 999 行 = 1000 行
                    break
            
            return {
                'exists': True,
                'size_mb': size_mb,
                'size_gb': size_gb,
                'readable': True,
                'format': 'gzip' if is_gzip else 'json',
                'sample_keys': list(data.keys()),
                'estimated_lines': count if count < 1000 else f"{count}+"
            }
    except Exception as e:
        return {
            'exists': True,
            'size_mb': size_mb,
            'size_gb': size_gb,
            'readable': False,
            'error': str(e)
        }

def main():
    print("=" * 70)
    print("📊 資料集驗證工具")
    print("=" * 70)
    
    # 定義要檢查的資料集
    datasets = {
        "California POI (壓縮)": "datasets/meta-California.json.gz",
        "California 評論 (壓縮)": "datasets/review-California.json.gz",
        "測試 POI (未壓縮)": "datasets/meta-other.json",
        "測試 POI (壓縮)": "datasets/meta-other.json.gz",
        "測試評論 (未壓縮)": "datasets/review-other.json",
        "測試評論 (壓縮)": "datasets/review-other.json.gz",
    }
    
    found_datasets = []
    missing_datasets = []
    
    for name, filepath in datasets.items():
        print(f"\n檢查: {name}")
        print(f"路徑: {filepath}")
        
        result = check_file(filepath)
        
        if result is None:
            print(f"  ❌ 檔案不存在")
            missing_datasets.append((name, filepath))
        elif not result['readable']:
            print(f"  ⚠️  檔案存在但無法讀取")
            print(f"  大小: {result['size_mb']:.2f} MB")
            print(f"  錯誤: {result.get('error', 'Unknown')}")
        else:
            print(f"  ✅ 檔案正常")
            if result['size_gb'] >= 1:
                print(f"  大小: {result['size_gb']:.2f} GB")
            else:
                print(f"  大小: {result['size_mb']:.2f} MB")
            print(f"  格式: {result['format']}")
            print(f"  估計行數: {result['estimated_lines']}")
            print(f"  資料欄位: {', '.join(result['sample_keys'][:10])}")
            found_datasets.append((name, filepath, result))
    
    # 總結
    print("\n" + "=" * 70)
    print("📋 檢查總結")
    print("=" * 70)
    print(f"✅ 找到 {len(found_datasets)} 個可用資料集")
    print(f"❌ 缺少 {len(missing_datasets)} 個資料集")
    
    if found_datasets:
        print("\n可用資料集：")
        for name, filepath, result in found_datasets:
            size_str = f"{result['size_gb']:.2f} GB" if result['size_gb'] >= 1 else f"{result['size_mb']:.2f} MB"
            print(f"  • {name}: {filepath} ({size_str})")
    
    if missing_datasets:
        print("\n缺少的資料集：")
        for name, filepath in missing_datasets:
            print(f"  • {name}: {filepath}")
    
    # 推薦使用的資料集
    print("\n" + "=" * 70)
    print("💡 使用建議")
    print("=" * 70)
    
    # 檢查是否有 California 資料集
    has_california = any('California' in name for name, _, _ in found_datasets)
    has_other = any('other' in filepath for _, filepath, _ in found_datasets)
    
    if has_california:
        print("🎯 推薦使用完整 California 資料集進行訓練：")
        print("   python train_model.py \\")
        print("     --meta-path datasets/meta-California.json.gz \\")
        print("     --review-path datasets/review-California.json.gz \\")
        print("     --max-pois 50000 \\")
        print("     --max-reviews 500000 \\")
        print("     --epochs 20")
    
    if has_other:
        print("\n🧪 快速測試使用小型資料集：")
        print("   python train_model.py \\")
        print("     --meta-path datasets/meta-other.json \\")
        print("     --review-path datasets/review-other.json \\")
        print("     --max-pois 1000 \\")
        print("     --max-reviews 5000 \\")
        print("     --epochs 5")
    
    if not has_california and not has_other:
        print("\n⚠️  未找到任何可用的資料集！")
        print("請確認資料集檔案已放置在 datasets/ 目錄中。")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
