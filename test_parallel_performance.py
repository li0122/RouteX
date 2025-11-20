#!/usr/bin/env python3
"""
並行處理效果測試
比較串行vs並行處理的性能差異
"""

import time
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def serial_negative_sampling_test(user_count=10000, poi_count=50000):
    """串行負樣本生成測試"""
    print(f"串行測試: {user_count:,} 用戶 × {poi_count:,} POI")
    
    # 模擬數據
    all_poi_ids = list(range(poi_count))
    user_interacted = {}
    
    for user_id in range(user_count):
        # 每個用戶隨機互動5-20個POI
        num_interactions = np.random.randint(5, 21)
        interacted = set(np.random.choice(all_poi_ids, num_interactions, replace=False))
        user_interacted[user_id] = interacted
    
    start_time = time.time()
    
    negative_samples = []
    for user_id, interacted in user_interacted.items():
        available_count = len(all_poi_ids) - len(interacted)
        target_negatives = min(10, available_count)  # 每用戶10個負樣本
        
        if target_negatives > 0:
            sample_size = min(target_negatives * 3, len(all_poi_ids))
            candidates = np.random.choice(all_poi_ids, sample_size, replace=False)
            neg_pois = [poi for poi in candidates if poi not in interacted][:target_negatives]
            
            for poi_id in neg_pois:
                negative_samples.append((user_id, poi_id, 0))
    
    elapsed_time = time.time() - start_time
    print(f"  串行處理耗時: {elapsed_time:.2f}秒")
    print(f"  生成負樣本數: {len(negative_samples):,}")
    print(f"  處理速度: {user_count/elapsed_time:.1f} 用戶/秒")
    
    return elapsed_time, len(negative_samples)

def process_user_batch_test(args):
    """並行處理的worker函數"""
    batch_users, user_interacted_batch, all_poi_ids = args
    
    batch_negatives = []
    
    for user_id in batch_users:
        if user_id not in user_interacted_batch:
            continue
            
        interacted = user_interacted_batch[user_id]
        available_count = len(all_poi_ids) - len(interacted)
        target_negatives = min(10, available_count)
        
        if target_negatives > 0:
            sample_size = min(target_negatives * 3, len(all_poi_ids))
            candidates = np.random.choice(all_poi_ids, sample_size, replace=False)
            neg_pois = [poi for poi in candidates if poi not in interacted][:target_negatives]
            
            for poi_id in neg_pois:
                batch_negatives.append((user_id, poi_id, 0))
    
    return batch_negatives

def parallel_negative_sampling_test(user_count=10000, poi_count=50000, max_workers=None):
    """並行負樣本生成測試"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    print(f"並行測試: {user_count:,} 用戶 × {poi_count:,} POI (worker數: {max_workers})")
    
    # 模擬數據
    all_poi_ids = list(range(poi_count))
    user_interacted = {}
    
    for user_id in range(user_count):
        num_interactions = np.random.randint(5, 21)
        interacted = set(np.random.choice(all_poi_ids, num_interactions, replace=False))
        user_interacted[user_id] = interacted
    
    start_time = time.time()
    
    # 準備並行任務
    batch_size = max(50, user_count // (max_workers * 4))
    tasks = []
    
    for i in range(0, user_count, batch_size):
        batch_users = list(range(i, min(i + batch_size, user_count)))
        user_interacted_batch = {uid: user_interacted[uid] for uid in batch_users if uid in user_interacted}
        
        task_args = (batch_users, user_interacted_batch, all_poi_ids)
        tasks.append(task_args)
    
    # 並行執行
    negative_samples = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_user_batch_test, task): i for i, task in enumerate(tasks)}
        
        for future in as_completed(future_to_task):
            try:
                batch_result = future.result()
                negative_samples.extend(batch_result)
            except Exception as e:
                print(f"並行任務失敗: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"  並行處理耗時: {elapsed_time:.2f}秒")
    print(f"  生成負樣本數: {len(negative_samples):,}")
    print(f"  處理速度: {user_count/elapsed_time:.1f} 用戶/秒")
    
    return elapsed_time, len(negative_samples)

def run_performance_test():
    """運行性能測試"""
    print("="*60)
    print("並行處理性能測試")
    print("="*60)
    
    # 測試配置
    test_configs = [
        (5000, 20000),   # 小規模
        (10000, 50000),  # 中規模  
        (20000, 100000), # 大規模
    ]
    
    for user_count, poi_count in test_configs:
        print(f"\\n測試規模: {user_count:,} 用戶 × {poi_count:,} POI")
        print("-" * 50)
        
        # 串行測試
        serial_time, serial_samples = serial_negative_sampling_test(user_count, poi_count)
        
        # 並行測試
        parallel_time, parallel_samples = parallel_negative_sampling_test(user_count, poi_count)
        
        # 性能比較
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"\\n性能比較:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  時間節省: {((serial_time - parallel_time) / serial_time * 100):.1f}%")
        
        if abs(serial_samples - parallel_samples) / max(serial_samples, 1) > 0.1:
            print(f"  ️ 樣本數差異較大: 串行={serial_samples:,}, 並行={parallel_samples:,}")
        else:
            print(f"   樣本數一致: {serial_samples:,}")

def check_system_resources():
    """檢查系統資源"""
    print("系統資源檢查:")
    print(f"  CPU核心數: {mp.cpu_count()}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  記憶體: {memory.total / (1024**3):.1f} GB (可用: {memory.available / (1024**3):.1f} GB)")
        print(f"  記憶體使用率: {memory.percent:.1f}%")
    except ImportError:
        print("  無法獲取記憶體資訊 (psutil未安裝)")
    
    print()

def main():
    print("RouteX 並行處理優化測試")
    print("="*60)
    
    check_system_resources()
    
    print("這個測試將比較串行vs並行處理的性能差異")
    print("測試將使用模擬數據，不會影響實際訓練數據")
    
    response = input("\\n是否開始測試? (y/N): ").strip().lower()
    if response == 'y':
        run_performance_test()
        
        print("\\n" + "="*60)
        print("測試完成！")
        print("如果並行處理顯示明顯加速，建議在實際訓練中啟用並行模式")
        print("使用命令: python train_model.py --parallel-workers 8 [其他參數]")
        print("或直接運行: ./train_parallel.sh")
    else:
        print("測試已取消")

if __name__ == "__main__":
    main()