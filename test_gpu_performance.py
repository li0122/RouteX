#!/usr/bin/env python3
"""
GPU加速負樣本生成性能測試
比較CPU vs GPU處理速度
"""

import time
import torch
import numpy as np
from typing import Dict, List

def cpu_negative_sampling_test(user_count=10000, poi_count=50000):
    """CPU負樣本生成測試"""
    print(f"CPU測試: {user_count:,} 用戶 × {poi_count:,} POI")
    
    # 模擬數據
    all_poi_ids = list(range(poi_count))
    user_interacted = {}
    
    for user_id in range(user_count):
        num_interactions = np.random.randint(5, 21)
        interacted = set(np.random.choice(all_poi_ids, num_interactions, replace=False))
        user_interacted[user_id] = interacted
    
    start_time = time.time()
    
    negative_samples = []
    for user_id, interacted in user_interacted.items():
        available_pois = [poi for poi in all_poi_ids if poi not in interacted]
        target_negatives = min(10, len(available_pois))
        
        if target_negatives > 0:
            neg_pois = np.random.choice(available_pois, target_negatives, replace=False)
            for poi_id in neg_pois:
                negative_samples.append((user_id, poi_id, 0))
    
    elapsed_time = time.time() - start_time
    print(f"  CPU處理耗時: {elapsed_time:.2f}秒")
    print(f"  生成負樣本數: {len(negative_samples):,}")
    print(f"  處理速度: {user_count/elapsed_time:.1f} 用戶/秒")
    
    return elapsed_time, len(negative_samples)

def gpu_negative_sampling_test(user_count=10000, poi_count=50000, device=None, batch_size=5000):
    """GPU負樣本生成測試"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("GPU不可用，跳過GPU測試")
        return None, None
    
    print(f"GPU測試: {user_count:,} 用戶 × {poi_count:,} POI (設備: {device})")
    
    # 模擬數據
    all_poi_ids = list(range(poi_count))
    user_interacted = {}
    
    for user_id in range(user_count):
        num_interactions = np.random.randint(5, 21)
        interacted = set(np.random.choice(all_poi_ids, num_interactions, replace=False))
        user_interacted[user_id] = interacted
    
    start_time = time.time()
    
    # 將POI轉為Tensor
    poi_tensor = torch.tensor(all_poi_ids, dtype=torch.long, device=device)
    num_pois = len(all_poi_ids)
    
    negative_samples = []
    user_list = list(user_interacted.keys())
    
    try:
        # 分批處理
        for batch_start in range(0, len(user_list), batch_size):
            batch_end = min(batch_start + batch_size, len(user_list))
            batch_users = user_list[batch_start:batch_end]
            batch_size_actual = len(batch_users)
            
            # 建立互動矩陣
            interaction_matrix = torch.zeros(batch_size_actual, num_pois, dtype=torch.bool, device=device)
            
            for i, user_id in enumerate(batch_users):
                if user_id in user_interacted:
                    interacted_pois = list(user_interacted[user_id])
                    if interacted_pois:
                        poi_indices = [all_poi_ids.index(poi) for poi in interacted_pois if poi in all_poi_ids]
                        if poi_indices:
                            interaction_matrix[i, poi_indices] = True
            
            # GPU向量化運算
            available_matrix = ~interaction_matrix
            available_counts = available_matrix.sum(dim=1)
            
            # 為每個用戶生成負樣本
            for i, user_id in enumerate(batch_users):
                if available_counts[i] <= 0:
                    continue
                
                available_mask = available_matrix[i]
                available_poi_indices = torch.nonzero(available_mask, as_tuple=True)[0]
                
                if len(available_poi_indices) == 0:
                    continue
                
                # 隨機採樣
                num_samples = min(10, len(available_poi_indices))
                if num_samples > 0:
                    perm = torch.randperm(len(available_poi_indices), device=device)[:num_samples]
                    selected_poi_indices = available_poi_indices[perm]
                    selected_pois = poi_tensor[selected_poi_indices].cpu().numpy().tolist()
                    
                    for poi_id in selected_pois:
                        negative_samples.append((user_id, poi_id, 0))
            
            # 清理GPU記憶體
            del interaction_matrix, available_matrix, available_counts
            torch.cuda.empty_cache()
    
    except torch.cuda.OutOfMemoryError:
        print("  GPU記憶體不足，減小批次大小重試...")
        torch.cuda.empty_cache()
        return gpu_negative_sampling_test(user_count, poi_count, device, batch_size // 2)
    
    elapsed_time = time.time() - start_time
    print(f"  GPU處理耗時: {elapsed_time:.2f}秒")
    print(f"  生成負樣本數: {len(negative_samples):,}")
    print(f"  處理速度: {user_count/elapsed_time:.1f} 用戶/秒")
    print(f"  GPU記憶體峰值: {torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")
    
    return elapsed_time, len(negative_samples)

def simple_gpu_test(user_count=1000, poi_count=5000):
    """簡單的GPU測試，用於排除問題"""
    print(f"\n簡單 GPU 測試: {user_count} 用戶 x {poi_count} POI")
    
    if not torch.cuda.is_available():
        print("GPU 不可用")
        return False
    
    try:
        device = torch.device('cuda')
        print(f"  使用設備: {device}")
        
        # 創建簡單張量
        print("  步驟1: 創建測試張量...")
        a = torch.randn(user_count, poi_count, device=device)
        
        print("  步驟2: 執行矩陣運算...")
        b = torch.matmul(a, a.T)
        
        print("  步驟3: 布爾運算...")
        mask = a > 0
        result = torch.sum(mask, dim=1)
        
        print("  步驟4: 清理記憶體...")
        del a, b, mask, result
        torch.cuda.empty_cache()
        
        print("  ✓ 簡單 GPU 測試成功")
        return True
        
    except Exception as e:
        print(f"  ✗ 簡單 GPU 測試失敗: {e}")
        torch.cuda.empty_cache()
        return False


def run_gpu_performance_test():
    """運行GPU性能測試"""
    print("="*60)
    print("GPU加速負樣本生成性能測試")
    print("="*60)
    
    # 先運行簡單GPU測試
    print("首先運行簡單GPU測試...")
    if not simple_gpu_test():
        print("❌ 簡單GPU測試失敗，跳過完整測試")
        return
    
    # 檢查GPU狀態
    if torch.cuda.is_available():
        print(f"GPU設備: {torch.cuda.get_device_name()}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("⚠️ GPU不可用，只能進行CPU測試")
    
    print()
    
    # 測試配置
    test_configs = [
        (5000, 20000),   # 小規模
        (10000, 50000),  # 中規模
        (20000, 100000), # 大規模
    ]
    
    for user_count, poi_count in test_configs:
        print(f"測試規模: {user_count:,} 用戶 × {poi_count:,} POI")
        print("-" * 50)
        
        # CPU測試
        cpu_time, cpu_samples = cpu_negative_sampling_test(user_count, poi_count)
        
        # GPU測試
        if torch.cuda.is_available():
            # 重置GPU記憶體
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            gpu_time, gpu_samples = gpu_negative_sampling_test(user_count, poi_count)
            
            if gpu_time is not None:
                # 性能比較
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"\n性能比較:")
                print(f"  GPU加速比: {speedup:.2f}x")
                print(f"  時間節省: {((cpu_time - gpu_time) / cpu_time * 100):.1f}%")
                
                if abs(cpu_samples - gpu_samples) / max(cpu_samples, 1) > 0.1:
                    print(f"  ⚠️ 樣本數差異: CPU={cpu_samples:,}, GPU={gpu_samples:,}")
                else:
                    print(f"  ✓ 樣本數一致: {cpu_samples:,}")
            else:
                print("GPU測試失敗")
        else:
            print("跳過GPU測試 (GPU不可用)")
        
        print()

def check_gpu_capability():
    """檢查GPU能力"""
    print("GPU能力檢查:")
    
    if not torch.cuda.is_available():
        print("  ❌ CUDA不可用")
        return False
    
    print(f"  ✓ CUDA可用 (版本: {torch.version.cuda})")
    print(f"  ✓ GPU數量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  ✓ GPU {i}: {props.name}")
        print(f"    記憶體: {props.total_memory / 1024**3:.1f} GB")
        print(f"    計算能力: {props.major}.{props.minor}")
        print(f"    多處理器: {props.multi_processor_count}")
    
    # 測試基本GPU操作
    try:
        device = torch.device('cuda')
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"  ✓ GPU運算測試通過")
        del test_tensor, result
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ❌ GPU運算測試失敗: {e}")
        return False

def main():
    print("RouteX GPU加速負樣本生成測試")
    print("="*60)
    
    # 檢查GPU能力
    gpu_available = check_gpu_capability()
    print()
    
    if not gpu_available:
        print("GPU不可用或測試失敗，建議檢查CUDA安裝")
        return
    
    print("這個測試將比較CPU vs GPU處理負樣本生成的性能差異")
    response = input("是否開始測試? (y/N): ").strip().lower()
    
    if response == 'y':
        run_gpu_performance_test()
        
        print("="*60)
        print("測試完成！")
        print("如果GPU顯示明顯加速，建議在實際訓練中啟用GPU模式")
        print("使用命令: python train_model.py --use-gpu-sampling [其他參數]")
        print("或直接運行: ./train_gpu_accelerated.sh")
    else:
        print("測試已取消")

if __name__ == "__main__":
    main()