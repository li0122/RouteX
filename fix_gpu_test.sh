#!/bin/bash

# GPU測試問題修復腳本

echo "=============================="
echo "GPU測試問題診斷和修復"
echo "=============================="

echo "1. 檢查CUDA環境..."
nvidia-smi
echo ""

echo "2. 檢查PyTorch安裝..."
python -c "
try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA版本: {torch.version.cuda}')
        print(f'GPU設備: {torch.cuda.get_device_name()}')
    else:
        print('CUDA不可用，可能需要重新安裝PyTorch')
except ImportError:
    print('PyTorch未安裝')
    exit(1)
"

echo ""
echo "3. 運行快速GPU診斷..."
python quick_gpu_test.py

echo ""
echo "4. 如果快速診斷成功，運行簡化測試..."

# 創建簡化的GPU測試
python -c "
import torch
import time
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('運行簡化GPU負樣本生成測試...')
    
    # 小規模測試
    users = 1000
    pois = 5000
    
    print(f'測試規模: {users} 用戶 × {pois} POI')
    
    try:
        # 模擬數據
        start_time = time.time()
        
        # 創建互動矩陣
        interaction_matrix = torch.zeros(users, pois, dtype=torch.bool, device=device)
        
        # 隨機填充
        for i in range(users):
            num_interactions = np.random.randint(5, 21)
            poi_indices = np.random.choice(pois, num_interactions, replace=False)
            interaction_matrix[i, poi_indices] = True
        
        # 計算可用POI
        available_matrix = ~interaction_matrix
        available_counts = available_matrix.sum(dim=1)
        
        # 為每個用戶生成負樣本
        negative_samples = []
        for i in range(users):
            if available_counts[i] > 0:
                available_mask = available_matrix[i]
                available_indices = torch.nonzero(available_mask, as_tuple=True)[0]
                
                if len(available_indices) > 0:
                    num_samples = min(10, len(available_indices))
                    perm = torch.randperm(len(available_indices), device=device)[:num_samples]
                    selected_indices = available_indices[perm]
                    selected_pois = selected_indices.cpu().numpy().tolist()
                    
                    for poi in selected_pois:
                        negative_samples.append((i, poi, 0))
        
        elapsed = time.time() - start_time
        
        print(f'✓ 簡化測試成功!')
        print(f'  耗時: {elapsed:.2f}秒')
        print(f'  生成負樣本數: {len(negative_samples):,}')
        print(f'  處理速度: {users/elapsed:.1f} 用戶/秒')
        
        # 清理
        del interaction_matrix, available_matrix, available_counts
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f'✗ 簡化測試失敗: {e}')
        torch.cuda.empty_cache()
else:
    print('GPU不可用')
"

echo ""
echo "=============================="
echo "診斷完成"
echo "=============================="
echo ""
echo "如果以上測試都成功，問題可能是:"
echo "1. 原始測試的批次大小太大"
echo "2. 矩陣初始化時間過長"
echo "3. 需要添加更多進度提示"
echo ""
echo "建議使用更小的測試規模重新運行:"
echo "python -c \"
from test_gpu_performance import gpu_negative_sampling_test
gpu_negative_sampling_test(1000, 5000, batch_size=500)
\""