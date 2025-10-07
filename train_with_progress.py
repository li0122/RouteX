#!/usr/bin/env python3
"""
訓練腳本包裝器 - 提供更清晰的進度顯示
"""

import sys
import subprocess
import time
from datetime import datetime

def print_header(text):
    """打印標題"""
    print("\n" + "="*60)
    print(text)
    print("="*60)

def print_step(step_num, text):
    """打印步驟"""
    print(f"\n{'='*60}")
    print(f"步驟 {step_num}: {text}")
    print("="*60)

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練旅行推薦模型（帶進度顯示）')
    parser.add_argument('--meta-path', required=True, help='POI 元數據路徑')
    parser.add_argument('--review-path', required=True, help='評論數據路徑')
    parser.add_argument('--max-pois', type=int, help='最大 POI 數量')
    parser.add_argument('--max-reviews', type=int, help='最大評論數量')
    parser.add_argument('--epochs', type=int, default=10, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--memory-efficient', action='store_true', help='啟用記憶體高效模式')
    parser.add_argument('--use-sharding', action='store_true', help='使用資料分片')
    parser.add_argument('--shard-size', type=int, default=100000, help='分片大小')
    
    args = parser.parse_args()
    
    # 打印訓練配置
    print_header("旅行推薦模型訓練")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n配置:")
    print(f"  - POI 資料: {args.meta_path}")
    print(f"  - 評論資料: {args.review_path}")
    print(f"  - 最大 POIs: {args.max_pois if args.max_pois else '全部'}")
    print(f"  - 最大評論: {args.max_reviews if args.max_reviews else '全部'}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 學習率: {args.learning_rate}")
    print(f"  - 記憶體優化: {'是' if args.memory_efficient else '否'}")
    print(f"  - 資料分片: {'是' if args.use_sharding else '否'}")
    
    # 構建命令
    cmd = [
        sys.executable, 
        'train_model.py',
        '--meta-path', args.meta_path,
        '--review-path', args.review_path,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
    ]
    
    if args.max_pois:
        cmd.extend(['--max-pois', str(args.max_pois)])
    if args.max_reviews:
        cmd.extend(['--max-reviews', str(args.max_reviews)])
    if args.memory_efficient:
        cmd.append('--memory-efficient')
    if args.use_sharding:
        cmd.extend(['--use-sharding', '--shard-size', str(args.shard_size)])
    
    print("\n準備開始訓練...")
    print("(您將看到詳細的進度條和即時輸出)\n")
    time.sleep(2)
    
    # 執行訓練
    start_time = time.time()
    
    try:
        # 使用 subprocess 並即時顯示輸出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 即時輸出
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        exit_code = process.returncode
        
    except KeyboardInterrupt:
        print("\n\n訓練被用戶中斷")
        process.terminate()
        exit_code = 1
    
    # 計算耗時
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    # 打印總結
    print_header("訓練總結")
    if exit_code == 0:
        print("✓ 訓練成功完成！")
    else:
        print("✗ 訓練失敗")
    print(f"\n總耗時: {hours}小時 {minutes}分鐘 {seconds}秒")
    print(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
