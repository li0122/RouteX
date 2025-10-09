#!/usr/bin/env python3
"""
針對百萬級用戶的極速訓練配置
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))

from train_model import main

def create_ultra_fast_args():
    """創建極速訓練參數配置"""
    parser = argparse.ArgumentParser()
    
    # 極速數據參數
    parser.add_argument('--meta-path', type=str, default='datasets/meta-California.json.gz')
    parser.add_argument('--review-path', type=str, default='datasets/review-California.json.gz')
    parser.add_argument('--max-pois', type=int, default=5000)     # 大幅減少
    parser.add_argument('--max-reviews', type=int, default=100000) # 大幅減少
    parser.add_argument('--negative-ratio', type=int, default=1)   # 降低到1:1
    
    # 啟用所有優化
    parser.add_argument('--memory-efficient', action='store_true', default=True)
    parser.add_argument('--use-sharding', action='store_true', default=False)
    parser.add_argument('--shard-size', type=int, default=50000)
    parser.add_argument('--max-samples-in-memory', type=int, default=20000)
    
    # 極簡模型參數
    parser.add_argument('--embedding-dim', type=int, default=8)   # 大幅減少
    parser.add_argument('--bottom-mlp-dims', type=int, nargs='+', default=[32, 16])
    parser.add_argument('--top-mlp-dims', type=int, nargs='+', default=[64, 32, 16])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 快速訓練參數
    parser.add_argument('--epochs', type=int, default=2)          # 只訓練2個epoch
    parser.add_argument('--batch-size', type=int, default=1024)   # 大批次
    parser.add_argument('--learning-rate', type=float, default=0.01)  # 高學習率
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    
    # 輸出參數
    parser.add_argument('--checkpoint-path', type=str, default='models/travel_dlrm_ultra_fast.pth')
    parser.add_argument('--processor-path', type=str, default='models/poi_processor_ultra_fast.pkl')
    
    return parser.parse_args([])

if __name__ == "__main__":
    print("="*60)
    print("RouteX 極速訓練模式")
    print("專門針對百萬級用戶數據優化")
    print("="*60)
    
    args = create_ultra_fast_args()
    
    print("極速配置:")
    print(f"  最大POI數: {args.max_pois:,}")
    print(f"  最大評論數: {args.max_reviews:,}")
    print(f"  負樣本比例: {args.negative_ratio}:1")
    print(f"  Embedding維度: {args.embedding_dim}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  訓練Epochs: {args.epochs}")
    print("="*60)
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n訓練被用戶中斷")
    except Exception as e:
        print(f"\n\n訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()