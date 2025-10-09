#!/usr/bin/env python3
"""
快速GPU診斷測試
用於快速檢測GPU是否正常工作
"""

import torch
import time
import numpy as np

def check_gpu_basic():
    """檢查GPU基本功能"""
    print("=== GPU基本檢查 ===")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，請檢查CUDA安裝")
        return False
    
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU數量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  記憶體: {props.total_memory / 1024**3:.1f} GB")
        print(f"  計算能力: {props.major}.{props.minor}")
    
    return True

def test_gpu_basic_ops():
    """測試GPU基本運算"""
    print("\n=== GPU基本運算測試 ===")
    
    try:
        device = torch.device('cuda')
        print(f"使用設備: {device}")
        
        # 測試1: 簡單張量創建
        print("測試1: 張量創建...")
        a = torch.randn(1000, 1000, device=device)
        print("  ✓ 張量創建成功")
        
        # 測試2: 矩陣乘法
        print("測試2: 矩陣乘法...")
        start_time = time.time()
        c = torch.matmul(a, a)
        torch.cuda.synchronize()  # 確保GPU操作完成
        elapsed = time.time() - start_time
        print(f"  ✓ 矩陣乘法成功 (耗時: {elapsed:.3f}秒)")
        
        # 測試3: 布爾運算
        print("測試3: 布爾運算...")
        mask = a > 0
        result = torch.sum(mask, dim=1)
        print("  ✓ 布爾運算成功")
        
        # 測試4: 記憶體管理
        print("測試4: 記憶體管理...")
        print(f"  使用記憶體: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        del a, c, mask, result
        torch.cuda.empty_cache()
        print(f"  清理後: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print("  ✓ 記憶體管理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU運算失敗: {e}")
        return False

def test_gpu_large_matrix():
    """測試GPU大型矩陣運算"""
    print("\n=== GPU大型矩陣測試 ===")
    
    try:
        device = torch.device('cuda')
        
        # 模擬負樣本生成的矩陣運算
        users = 5000
        pois = 20000
        
        print(f"測試規模: {users} 用戶 × {pois} POI")
        
        # 創建布爾矩陣 (模擬用戶-POI互動)
        print("步驟1: 創建互動矩陣...")
        interaction_matrix = torch.zeros(users, pois, dtype=torch.bool, device=device)
        
        # 隨機填充一些True值
        print("步驟2: 隨機填充互動數據...")
        for i in range(users):
            num_interactions = np.random.randint(5, 21)
            poi_indices = np.random.choice(pois, num_interactions, replace=False)
            interaction_matrix[i, poi_indices] = True
        
        # 計算可用POI (負樣本候選)
        print("步驟3: 計算可用POI...")
        start_time = time.time()
        available_matrix = ~interaction_matrix
        available_counts = available_matrix.sum(dim=1)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"  ✓ 計算完成 (耗時: {elapsed:.3f}秒)")
        print(f"  平均可用POI數: {available_counts.float().mean().item():.1f}")
        
        # 清理
        del interaction_matrix, available_matrix, available_counts
        torch.cuda.empty_cache()
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU記憶體不足")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"❌ 大型矩陣測試失敗: {e}")
        torch.cuda.empty_cache()
        return False

def test_gpu_performance():
    """GPU性能測試"""
    print("\n=== GPU性能測試 ===")
    
    sizes = [
        (1000, 5000),
        (2000, 10000),
        (5000, 20000)
    ]
    
    for users, pois in sizes:
        print(f"\n測試規模: {users} × {pois}")
        
        try:
            device = torch.device('cuda')
            
            start_time = time.time()
            
            # 創建矩陣
            matrix = torch.randn(users, pois, device=device)
            
            # 執行運算
            result = torch.matmul(matrix, matrix.T)
            mask = result > 0
            counts = mask.sum(dim=1)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            print(f"  耗時: {elapsed:.3f}秒")
            print(f"  吞吐量: {users * pois / elapsed / 1e6:.2f} M操作/秒")
            
            del matrix, result, mask, counts
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  失敗: {e}")
            torch.cuda.empty_cache()

def main():
    print("RouteX GPU快速診斷")
    print("=" * 50)
    
    # 基本檢查
    if not check_gpu_basic():
        return
    
    # 基本運算測試
    if not test_gpu_basic_ops():
        print("❌ 基本GPU運算失敗，請檢查CUDA安裝")
        return
    
    # 大型矩陣測試
    if not test_gpu_large_matrix():
        print("❌ 大型矩陣測試失敗，可能是記憶體不足")
        return
    
    # 性能測試
    test_gpu_performance()
    
    print("\n" + "=" * 50)
    print("✓ GPU診斷完成！GPU功能正常")
    print("現在可以嘗試完整的GPU加速測試")

if __name__ == "__main__":
    main()