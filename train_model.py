"""
訓練旅行推薦模型
支援大資料集的記憶體高效訓練
支援多進程並行處理優化
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import gc

import time
import os
import torch.utils.data
import torch.cuda
from torch.utils.data import TensorDataset

from dlrm_model import create_travel_dlrm, DLRMLoss
from data_processor import load_and_process_data, POIDataProcessor, ReviewDataProcessor





def gpu_accelerated_negative_sampling(user_interacted, all_poi_ids, negative_ratio, device=None, batch_size=10000):
    """
    GPU加速的負樣本生成
    使用CUDA進行大規模並行計算
    """
    if device is None:
        device = torch.device('cuda')
    
    print(f"  使用GPU加速: {device}")
    
    # 創建POI ID到索引的映射
    poi_to_idx = {poi_id: idx for idx, poi_id in enumerate(all_poi_ids)}
    num_pois = len(all_poi_ids)
    
    print(f"  GPU記憶體狀態: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB / {torch.cuda.memory_reserved(device)/1024**3:.2f}GB")
    
    negative_samples = []
    user_list = list(user_interacted.keys())
    
    # 分批處理以節省GPU記憶體
    for batch_start in range(0, len(user_list), batch_size):
        batch_end = min(batch_start + batch_size, len(user_list))
        batch_users = user_list[batch_start:batch_end]
        
        print(f"    GPU批次: {batch_start//batch_size + 1}/{(len(user_list) + batch_size - 1)//batch_size} "
              f"(用戶 {batch_start+1}-{batch_end})")
        
        # 為這個批次的用戶建立互動矩陣
        batch_size_actual = len(batch_users)
        interaction_matrix = torch.zeros(batch_size_actual, num_pois, dtype=torch.bool, device=device)
        
        # 填充互動矩陣
        for i, user_id in enumerate(batch_users):
            if user_id in user_interacted:
                interacted_pois = list(user_interacted[user_id])
                if interacted_pois:
                    # 將POI ID轉換為索引
                    poi_indices = [poi_to_idx[poi] for poi in interacted_pois if poi in poi_to_idx]
                    if poi_indices:
                        interaction_matrix[i, poi_indices] = True
        
        # GPU上的向量化運算
        # 計算可用POI矩陣 (未互動的POI)
        available_matrix = ~interaction_matrix  # shape: (batch_size, num_pois)
        
        # 計算每個用戶的可用POI數量
        available_counts = available_matrix.sum(dim=1)  # shape: (batch_size,)
        
        # 計算每個用戶的互動POI數量
        interaction_counts = interaction_matrix.sum(dim=1)  # shape: (batch_size,)
        
        # 計算目標負樣本數量
        target_negatives = torch.minimum(
            negative_ratio * interaction_counts,
            torch.minimum(available_counts, torch.tensor(50, device=device))  # 最多50個負樣本
        )
        
        # 為每個用戶生成負樣本
        for i, user_id in enumerate(batch_users):
            if target_negatives[i] <= 0:
                continue
            
            # 獲取當前用戶的可用POI
            available_mask = available_matrix[i]  # shape: (num_pois,)
            available_poi_indices = torch.nonzero(available_mask, as_tuple=True)[0]
            
            if len(available_poi_indices) == 0:
                continue
            
            # GPU上的隨機採樣
            num_samples = min(int(target_negatives[i]), len(available_poi_indices))
            if num_samples > 0:
                # 隨機排列可用POI索引
                perm = torch.randperm(len(available_poi_indices), device=device)[:num_samples]
                selected_poi_indices = available_poi_indices[perm]
                
                # 轉換回原POI ID
                selected_pois = [all_poi_ids[idx.item()] for idx in selected_poi_indices]
                
                # 添加到結果
                for poi_id in selected_pois:
                    negative_samples.append((user_id, poi_id, 0))
        
        # 清理GPU記憶體
        del interaction_matrix, available_matrix, available_counts, interaction_counts, target_negatives
        torch.cuda.empty_cache()
        
        if (batch_start // batch_size + 1) % 5 == 0:
            print(f"      目前負樣本數: {len(negative_samples):,}")
            print(f"      GPU記憶體: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB")
    
    print(f"  GPU加速完成! 生成負樣本數: {len(negative_samples):,}")
    return negative_samples


def pure_gpu_negative_sampling(user_interacted, all_poi_ids, negative_ratio, device=None):
    """
    純GPU負樣本生成
    強制使用GPU進行所有計算
    """
    user_count = len(user_interacted)
    poi_count = len(all_poi_ids)
    
    print(f"  數據規模: {user_count:,} 用戶 × {poi_count:,} POI")
    print(f"  使用純GPU模式處理")
    
    if device is None:
        device = torch.device('cuda')
    
    # 直接使用GPU加速處理
    start_time = time.time()
    result = gpu_accelerated_negative_sampling(user_interacted, all_poi_ids, negative_ratio, device)
    gpu_time = time.time() - start_time
    
    print(f"  GPU處理耗時: {gpu_time:.2f}秒")
    return result





def load_data_in_shards(
    meta_path: str,
    review_path: str,
    max_pois: int = None,
    max_reviews: int = None,
    shard_size: int = 100000
) -> Tuple[POIDataProcessor, ReviewDataProcessor]:
    """
    分片載入資料以節省記憶體
    
    Args:
        meta_path: POI 資料路徑
        review_path: 評論資料路徑
        max_pois: 最大 POI 數量
        max_reviews: 最大評論數量
        shard_size: 每個分片大小
    """
    print(f"使用分片模式載入資料 (分片大小: {shard_size})...")
    
    # 載入 POI 資料（通常較小，可以一次載入）
    poi_processor = POIDataProcessor(meta_path)
    poi_processor.load_data(max_records=max_pois)
    poi_processor.preprocess()
    
    # 分片載入評論資料
    review_processor = ReviewDataProcessor(review_path)
    
    # 計算總評論數（用於進度顯示）
    import gzip
    is_gzip = review_path.endswith('.gz')
    open_func = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'
    
    print("分片載入評論資料...")
    reviews_loaded = 0
    shard_num = 0
    
    with open_func(review_path, mode, encoding='utf-8') as f:
        current_shard = []
        
        for i, line in enumerate(f):
            if max_reviews and reviews_loaded >= max_reviews:
                break
            
            try:
                review = json.loads(line.strip())
                current_shard.append(review)
                reviews_loaded += 1
                
                # 當分片滿了，處理並清空
                if len(current_shard) >= shard_size:
                    shard_num += 1
                    print(f"  處理分片 {shard_num} ({len(current_shard)} 條評論)...")
                    review_processor.reviews.extend(current_shard)
                    current_shard = []
                    gc.collect()  # 強制垃圾回收
                
                if reviews_loaded % 50000 == 0:
                    print(f"  已載入 {reviews_loaded} 條評論...")
                    
            except json.JSONDecodeError:
                continue
        
        # 處理最後一個分片
        if current_shard:
            shard_num += 1
            print(f"  處理最後分片 ({len(current_shard)} 條評論)...")
            review_processor.reviews.extend(current_shard)
    
    print(f"✓ 成功載入 {len(review_processor.reviews)} 條評論（{shard_num} 個分片）")
    
    # 預處理評論資料
    review_processor.preprocess()
    
    # 清理記憶體
    gc.collect()
    
    return poi_processor, review_processor


class TravelRecommendDataset(Dataset):
    """旅行推薦訓練數據集（支援記憶體高效模式）"""
    
    def __init__(
        self,
        poi_processor: POIDataProcessor,
        review_processor: ReviewDataProcessor,
        negative_ratio: int = 4,
        memory_efficient: bool = True,
        max_samples_in_memory: int = 50000,
        gpu_batch_size: int = 20000
    ):
        self.poi_processor = poi_processor
        self.review_processor = review_processor
        self.negative_ratio = negative_ratio
        self.memory_efficient = memory_efficient
        self.max_samples_in_memory = max_samples_in_memory
        self.gpu_batch_size = gpu_batch_size
        
        # 建立訓練樣本
        if memory_efficient:
            # 只儲存樣本索引，不儲存完整樣本
            self.samples = self._create_sample_indices()
        else:
            self.samples = self._create_samples()
    
    def _create_sample_indices(self) -> List[Tuple]:
        """創建訓練樣本索引（記憶體高效模式）"""
        print("使用記憶體高效模式...")
        samples = []
        
        # 正樣本: 用戶高評分的POI
        print("正在生成正樣本...")
        for user_id, reviews in self.review_processor.user_reviews.items():
            for review in reviews:
                rating = review.get('rating', 0)
                if rating >= 4.0:
                    poi_id = review.get('gmap_id')
                    if poi_id and poi_id in self.poi_processor.poi_index:
                        samples.append((user_id, poi_id, 1))
        
        print(f"✓ 生成 {len(samples)} 個正樣本索引")
        
        # 超高效負樣本生成策略
        print("正在生成負樣本（超高效索引模式）...")
        user_ids = list(self.review_processor.user_reviews.keys())
        all_poi_ids = list(self.poi_processor.poi_index.keys())
        
        print(f"  總用戶數: {len(user_ids):,}")
        print(f"  總POI數: {len(all_poi_ids):,}")
        print(f"  負樣本比例: {self.negative_ratio}:1")
        
        # 策略1: 智慧用戶採樣
        print("  步驟1: 篩選活躍用戶...")
        active_users = []
        for user_id in user_ids:
            reviews = self.review_processor.user_reviews[user_id]
            if len(reviews) >= 2:  # 至少2條評論
                ratings = [r.get('rating', 0) for r in reviews]
                if len(set(ratings)) > 1 or max(ratings) >= 4.0:
                    active_users.append(user_id)
        
        print(f"  篩選出 {len(active_users):,} 個活躍用戶")
        
        # 策略2: 控制計算量
        max_users = min(50000, len(active_users))
        if len(active_users) > max_users:
            print(f"  隨機採樣 {max_users:,} 個用戶...")
            active_users = np.random.choice(active_users, max_users, replace=False).tolist()
        
        # 策略3: 預計算用戶互動
        print("  步驟2: 預計算用戶互動記錄...")
        all_poi_set = set(all_poi_ids)
        user_interacted = {}
        
        for user_id in active_users:
            interacted = set(
                r.get('gmap_id') for r in self.review_processor.user_reviews[user_id]
                if r.get('gmap_id') in self.poi_processor.poi_index
            )
            if len(interacted) > 0:
                user_interacted[user_id] = interacted
        
        print(f"  有效用戶數: {len(user_interacted):,}")
        
        # 策略4: 純GPU高效生成
        print("  步驟3: 純GPU生成負樣本索引...")
        
        # 使用純GPU處理
        negative_samples = pure_gpu_negative_sampling(
            user_interacted=user_interacted,
            all_poi_ids=all_poi_ids,
            negative_ratio=self.negative_ratio,
            device=torch.device('cuda')
        )
        
        print(f"\n✓ 成功生成 {len(negative_samples):,} 個負樣本索引")
        print(f"✓ 總樣本數: {len(samples) + len(negative_samples):,} (正樣本: {len(samples):,}, 負樣本: {len(negative_samples):,})")
        
        samples.extend(negative_samples)
        print("  正在打亂樣本順序...")
        np.random.shuffle(samples)
        print("✓ 樣本創建完成!")
        
        return samples
    
    def _create_samples(self) -> List[Tuple]:
        """創建訓練樣本 (user, poi, label) - 超高效模式"""
        samples = []
        
        # 正樣本: 用戶高評分的POI
        print("正在生成正樣本...")
        for user_id, reviews in self.review_processor.user_reviews.items():
            for review in reviews:
                rating = review.get('rating', 0)
                
                # 4.0分以上視為正樣本
                if rating >= 4.0:
                    poi_id = review.get('gmap_id')
                    if poi_id and poi_id in self.poi_processor.poi_index:
                        samples.append((user_id, poi_id, 1))
        
        print(f"✓ 生成 {len(samples)} 個正樣本")
        
        # 超高效負樣本生成策略
        print("正在生成負樣本（超高效模式）...")
        user_ids = list(self.review_processor.user_reviews.keys())
        all_poi_ids = list(self.poi_processor.poi_index.keys())
        
        print(f"  總用戶數: {len(user_ids):,}")
        print(f"  總POI數: {len(all_poi_ids):,}")
        
        # 策略1: 智慧用戶採樣 - 只處理活躍用戶
        print("  步驟1: 篩選活躍用戶...")
        active_users = []
        for user_id in user_ids:
            reviews = self.review_processor.user_reviews[user_id]
            # 只保留有足夠互動且評分多樣的用戶
            if len(reviews) >= 2:  # 至少2條評論
                ratings = [r.get('rating', 0) for r in reviews]
                if len(set(ratings)) > 1 or max(ratings) >= 4.0:  # 有評分變化或有高評分
                    active_users.append(user_id)
        
        print(f"  篩選出 {len(active_users):,} 個活躍用戶 (原 {len(user_ids):,})")
        
        # 策略2: 進一步採樣以控制計算量
        max_users_for_negatives = min(50000, len(active_users))  # 最多處理5萬用戶
        if len(active_users) > max_users_for_negatives:
            print(f"  隨機採樣 {max_users_for_negatives:,} 個用戶進行負樣本生成...")
            active_users = np.random.choice(active_users, max_users_for_negatives, replace=False).tolist()
        
        # 策略3: 預計算所有用戶的互動POI
        print("  步驟2: 預計算用戶互動記錄...")
        all_poi_set = set(all_poi_ids)
        user_interacted = {}
        
        for user_id in active_users:
            interacted = set(
                r.get('gmap_id') for r in self.review_processor.user_reviews[user_id]
                if r.get('gmap_id') in self.poi_processor.poi_index
            )
            if len(interacted) > 0:
                user_interacted[user_id] = interacted
        
        print(f"  有效用戶數: {len(user_interacted):,}")
        
        # 策略4: 純GPU生成負樣本
        print("  步驟3: 純GPU生成負樣本...")
        
        negative_samples = pure_gpu_negative_sampling(
            user_interacted=user_interacted,
            all_poi_ids=all_poi_ids,
            negative_ratio=self.negative_ratio,
            device=torch.device('cuda')
        )
        
        print(f"\n✓ 成功生成 {len(negative_samples):,} 個負樣本")
        print(f"✓ 總樣本數: {len(samples) + len(negative_samples):,} (正樣本: {len(samples):,}, 負樣本: {len(negative_samples):,})")
        
        samples.extend(negative_samples)
        
        # 打亂
        print("  正在打亂樣本順序...")
        np.random.shuffle(samples)
        print("✓ 樣本創建完成!")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_id, poi_id, label = self.samples[idx]
        
        # 獲取用戶特徵
        user_profile = self.review_processor.get_user_profile(user_id)
        user_features = np.array([
            user_profile['avg_rating'] / 5.0,
            np.log1p(user_profile['num_reviews']),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        # 獲取POI特徵
        poi_data = self.poi_processor.pois[self.poi_processor.poi_index[poi_id]]
        poi_encoded = self.poi_processor.encode_poi(poi_data)
        
        # 路徑特徵 (訓練時使用隨機值)
        path_features = np.random.rand(4).astype(np.float32) * 0.1
        
        return {
            'user_continuous': user_features,
            'poi_continuous': poi_encoded['continuous'],
            'poi_category': poi_encoded['categorical']['category'],
            'poi_state': poi_encoded['categorical']['state'],
            'poi_price': poi_encoded['categorical']['price_level'],
            'path_continuous': path_features,
            'label': label
        }


def collate_fn(batch):
    """自定義 collate 函數"""
    user_continuous = torch.stack([torch.from_numpy(b['user_continuous']) for b in batch])
    poi_continuous = torch.stack([torch.from_numpy(b['poi_continuous']) for b in batch])
    path_continuous = torch.stack([torch.from_numpy(b['path_continuous']) for b in batch])
    
    poi_categorical = {
        'category': torch.tensor([b['poi_category'] for b in batch], dtype=torch.long),
        'state': torch.tensor([b['poi_state'] for b in batch], dtype=torch.long),
        'price_level': torch.tensor([b['poi_price'] for b in batch], dtype=torch.long)
    }
    
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
    
    return {
        'user_continuous': user_continuous,
        'user_categorical': {},
        'poi_continuous': poi_continuous,
        'poi_categorical': poi_categorical,
        'path_continuous': path_continuous,
        'labels': labels
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 使用 tqdm 並強制刷新輸出
    import sys
    for batch in tqdm(dataloader, desc="Training", file=sys.stdout, ncols=80):
        # 移動到設備
        user_continuous = batch['user_continuous'].to(device)
        poi_continuous = batch['poi_continuous'].to(device)
        path_continuous = batch['path_continuous'].to(device)
        poi_categorical = {k: v.to(device) for k, v in batch['poi_categorical'].items()}
        labels = batch['labels'].to(device)
        
        # 前向傳播
        output = model(
            user_continuous, {},
            poi_continuous, poi_categorical,
            path_continuous
        )
        
        scores = output['scores'].squeeze()
        
        # 計算損失 (BCE)
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """評估模型"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        import sys
        for batch in tqdm(dataloader, desc="Evaluating", file=sys.stdout, ncols=80):
            user_continuous = batch['user_continuous'].to(device)
            poi_continuous = batch['poi_continuous'].to(device)
            path_continuous = batch['path_continuous'].to(device)
            poi_categorical = {k: v.to(device) for k, v in batch['poi_categorical'].items()}
            labels = batch['labels'].to(device)
            
            output = model(
                user_continuous, {},
                poi_continuous, poi_categorical,
                path_continuous
            )
            
            scores = torch.sigmoid(output['scores'].squeeze())
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 計算指標
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
        
        predictions = (all_scores > 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(all_labels, all_scores),
            'accuracy': accuracy_score(all_labels, predictions),
            'precision': precision_score(all_labels, predictions),
            'recall': recall_score(all_labels, predictions)
        }
    except ImportError:
        print("⚠️ sklearn 未安裝，使用簡單指標計算")
        predictions = (all_scores > 0.5).astype(int)
        
        # 簡單指標計算
        tp = np.sum((all_labels == 1) & (predictions == 1))
        tn = np.sum((all_labels == 0) & (predictions == 0))
        fp = np.sum((all_labels == 0) & (predictions == 1))
        fn = np.sum((all_labels == 1) & (predictions == 0))
        
        accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'auc': 0.5,  # 預設值
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    return metrics


def main(args):
    """主訓練流程"""
    print("="*60)
    print("旅行推薦模型訓練")
    print("="*60)
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")
    
    # 檢查記憶體（可選）
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"系統記憶體: {ram_gb:.1f} GB (可用: {available_gb:.1f} GB)")
        
        # 根據記憶體決定模式
        memory_efficient = ram_gb < 16 or args.memory_efficient
    except ImportError:
        print("⚠️ psutil 未安裝，無法檢測記憶體")
        memory_efficient = args.memory_efficient
    
    if memory_efficient:
        print("⚠️  啟用記憶體高效模式")
    
    # 載入數據（分片模式）
    print("\n載入數據...")
    if args.use_sharding:
        print("使用資料分片模式...")
        poi_processor, review_processor = load_data_in_shards(
            meta_path=args.meta_path,
            review_path=args.review_path,
            max_pois=args.max_pois,
            max_reviews=args.max_reviews,
            shard_size=args.shard_size
        )
    else:
        poi_processor, review_processor = load_and_process_data(
            meta_path=args.meta_path,
            review_path=args.review_path,
            max_pois=args.max_pois,
            max_reviews=args.max_reviews
        )
    
    # 創建數據集
    print("\n創建訓練數據集...")
    
    # 檢查GPU設置
    if torch.cuda.is_available():
        print(f"✓ 使用GPU加速所有計算")
        print(f"  GPU設備: {torch.cuda.get_device_name()}")
        print(f"  GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        raise RuntimeError("⚠️ GPU不可用，無法執行")
    
    dataset = TravelRecommendDataset(
        poi_processor, 
        review_processor, 
        negative_ratio=args.negative_ratio,
        memory_efficient=memory_efficient,
        max_samples_in_memory=args.max_samples_in_memory,
        gpu_batch_size=args.gpu_batch_size
    )
    
    # 分割訓練/驗證集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")
    
    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # 創建模型
    print("\n" + "="*60)
    print("第 4 步：創建模型")
    print("="*60)
    
    poi_vocab_sizes = {
        'category': len(poi_processor.category_encoder),
        'state': len(poi_processor.state_encoder),
        'price_level': 5
    }
    
    print(f"\n模型配置:")
    print(f"  - Embedding 維度: {args.embedding_dim}")
    print(f"  - Bottom MLP: {args.bottom_mlp_dims}")
    print(f"  - Top MLP: {args.top_mlp_dims}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - 學習率: {args.learning_rate}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"\n正在初始化模型...")
    
    model = create_travel_dlrm(
        user_continuous_dim=10,
        poi_continuous_dim=8,
        path_continuous_dim=4,
        user_vocab_sizes={},
        poi_vocab_sizes=poi_vocab_sizes,
        embedding_dim=args.embedding_dim,
        bottom_mlp_dims=args.bottom_mlp_dims,
        top_mlp_dims=args.top_mlp_dims,
        dropout=args.dropout
    )
    
    model = model.to(device)
    
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 優化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # 訓練
    print("\n" + "="*60)
    print("第 5 步：開始訓練")
    print("="*60)
    print(f"總 Epochs: {args.epochs}")
    print(f"每個 Epoch 包含 {len(train_loader)} 個批次")
    print("\n" + "="*60 + "\n")
    
    import time
    training_start_time = time.time()
    best_auc = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # 訓練
        print(f"\n[訓練階段] 處理 {len(train_loader)} 個批次...")
        train_loss = train_epoch(model, train_loader, optimizer, None, device)
        print(f"\n✓ 訓練完成")
        print(f"  訓練損失: {train_loss:.4f}")
        
        # 驗證
        print(f"\n[驗證階段] 評估模型效能...")
        val_metrics = evaluate(model, val_loader, device)
        print(f"\n✓ 驗證完成")
        print(f"  驗證指標:")
        print(f"    - AUC: {val_metrics['auc']:.4f}")
        print(f"    - Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    - Precision: {val_metrics['precision']:.4f}")
        print(f"    - Recall: {val_metrics['recall']:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"\n  Epoch 耗時: {epoch_time:.1f} 秒")
        
        # 學習率調度
        scheduler.step(val_metrics['auc'])
        
        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'poi_vocab_sizes': poi_vocab_sizes
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"✓ 保存最佳模型 (AUC: {best_auc:.4f})")
    
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print("\n" + "="*60)
    print("訓練完成！")
    print("="*60)
    print(f"最佳驗證 AUC: {best_auc:.4f}")
    print(f"總訓練時間: {hours}h {minutes}m {seconds}s")
    print(f"模型已儲存至: {args.checkpoint_path}")
    print(f"結束時間: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 保存處理器
    poi_processor.save(args.processor_path)
    print(f"✓ 數據處理器已保存到 {args.processor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練旅行推薦模型")
    
    # 數據參數
    parser.add_argument('--meta-path', type=str, default='datasets/meta-California.json.gz')
    parser.add_argument('--review-path', type=str, default='datasets/review-California.json.gz')
    parser.add_argument('--max-pois', type=int, default=10000)
    parser.add_argument('--max-reviews', type=int, default=50000)
    parser.add_argument('--negative-ratio', type=int, default=4)
    
    # 記憶體管理參數
    parser.add_argument('--memory-efficient', action='store_true', help='啟用記憶體高效模式')
    parser.add_argument('--use-sharding', action='store_true', help='使用資料分片')
    parser.add_argument('--shard-size', type=int, default=100000, help='每個分片的大小')
    parser.add_argument('--max-samples-in-memory', type=int, default=50000, help='記憶體中最大樣本數')
    
    # 模型參數
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--bottom-mlp-dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--top-mlp-dims', type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    
    # GPU處理參數
    parser.add_argument('--gpu-batch-size', type=int, default=10000, help='GPU處理批次大小')
    
    # 輸出參數
    parser.add_argument('--checkpoint-path', type=str, default='models/travel_dlrm.pth')
    parser.add_argument('--processor-path', type=str, default='models/poi_processor.pkl')
    
    args = parser.parse_args()
    
    # 創建模型目錄
    Path('models').mkdir(exist_ok=True)
    
    main(args)
