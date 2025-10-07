"""
訓練旅行推薦模型
支援大資料集的記憶體高效訓練
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

from dlrm_model import create_travel_dlrm, DLRMLoss
from data_processor import load_and_process_data, POIDataProcessor, ReviewDataProcessor


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
        max_samples_in_memory: int = 50000
    ):
        self.poi_processor = poi_processor
        self.review_processor = review_processor
        self.negative_ratio = negative_ratio
        self.memory_efficient = memory_efficient
        self.max_samples_in_memory = max_samples_in_memory
        
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
        for user_id, reviews in self.review_processor.user_reviews.items():
            for review in reviews:
                rating = review.get('rating', 0)
                if rating >= 4.0:
                    poi_id = review.get('gmap_id')
                    if poi_id and poi_id in self.poi_processor.poi_index:
                        samples.append((user_id, poi_id, 1))
        
        print(f"生成 {len(samples)} 個正樣本索引")
        
        # 負樣本索引
        all_poi_ids = list(self.poi_processor.poi_index.keys())
        user_ids = list(self.review_processor.user_reviews.keys())
        
        # 為每個用戶記錄已互動的 POI
        user_interacted = {}
        for user_id in user_ids:
            user_interacted[user_id] = set(
                r.get('gmap_id') for r in self.review_processor.user_reviews[user_id]
            )
        
        # 生成負樣本（分批以節省記憶體）
        negative_samples = []
        batch_size = 1000
        
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i+batch_size]
            for user_id in batch_users:
                interacted = user_interacted[user_id]
                available_pois = [p for p in all_poi_ids if p not in interacted]
                
                if available_pois:
                    num_negatives = min(
                        self.negative_ratio * len(interacted),
                        len(available_pois)
                    )
                    neg_pois = np.random.choice(available_pois, num_negatives, replace=False)
                    for poi_id in neg_pois:
                        negative_samples.append((user_id, poi_id, 0))
        
        print(f"生成 {len(negative_samples)} 個負樣本索引")
        samples.extend(negative_samples)
        np.random.shuffle(samples)
        
        return samples
    
    def _create_samples(self) -> List[Tuple]:
        """創建訓練樣本 (user, poi, label) - 標準模式"""
        samples = []
        
        # 正樣本: 用戶高評分的POI
        for user_id, reviews in self.review_processor.user_reviews.items():
            for review in reviews:
                rating = review.get('rating', 0)
                
                # 4.0分以上視為正樣本
                if rating >= 4.0:
                    poi_id = review.get('gmap_id')
                    if poi_id and poi_id in self.poi_processor.poi_index:
                        samples.append((user_id, poi_id, 1))
        
        print(f"生成 {len(samples)} 個正樣本")
        
        # 負樣本: 隨機未互動的POI
        all_poi_ids = list(self.poi_processor.poi_index.keys())
        negative_samples = []
        
        for user_id in self.review_processor.user_reviews.keys():
            # 用戶已互動的POI
            interacted_pois = set(
                r.get('gmap_id') for r in self.review_processor.user_reviews[user_id]
            )
            
            # 隨機採樣未互動的POI
            available_pois = [p for p in all_poi_ids if p not in interacted_pois]
            
            if available_pois:
                num_negatives = min(
                    self.negative_ratio * len(interacted_pois),
                    len(available_pois)
                )
                
                neg_pois = np.random.choice(available_pois, num_negatives, replace=False)
                
                for poi_id in neg_pois:
                    negative_samples.append((user_id, poi_id, 0))
        
        print(f"生成 {len(negative_samples)} 個負樣本")
        
        samples.extend(negative_samples)
        
        # 打亂
        np.random.shuffle(samples)
        
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
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    predictions = (all_scores > 0.5).astype(int)
    
    metrics = {
        'auc': roc_auc_score(all_labels, all_scores),
        'accuracy': accuracy_score(all_labels, predictions),
        'precision': precision_score(all_labels, predictions),
        'recall': recall_score(all_labels, predictions)
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
    dataset = TravelRecommendDataset(
        poi_processor, 
        review_processor, 
        negative_ratio=args.negative_ratio,
        memory_efficient=memory_efficient,
        max_samples_in_memory=args.max_samples_in_memory
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
    parser.add_argument('--meta-path', type=str, default='datasets/meta-other.json')
    parser.add_argument('--review-path', type=str, default='datasets/review-other.json')
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
    
    # 輸出參數
    parser.add_argument('--checkpoint-path', type=str, default='models/travel_dlrm.pth')
    parser.add_argument('--processor-path', type=str, default='models/poi_processor.pkl')
    
    args = parser.parse_args()
    
    # 創建模型目錄
    Path('models').mkdir(exist_ok=True)
    
    main(args)
