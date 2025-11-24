"""
完整模型評估程式
計算 Precision@K, Recall@K, F1-Score, NDCG@K, AUC 等指標
支援 K = 1, 3, 10
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm

from dlrm_model import create_travel_dlrm
from data_processor import POIDataProcessor, ReviewDataProcessor


class RecommenderEvaluator:
    """推薦系統評估器"""
    
    def __init__(self, model, poi_processor, k_values: List[int] = [1, 3, 10]):
        """
        Args:
            model: 訓練好的 DLRM 模型
            poi_processor: POI 資料處理器
            k_values: 要評估的 K 值列表
        """
        self.model = model
        self.poi_processor = poi_processor
        self.k_values = sorted(k_values)
        self.device = next(model.parameters()).device
        
    def precision_at_k(self, relevant: List, recommended: List, k: int) -> float:
        """
        計算 Precision@K
        
        Args:
            relevant: 實際相關的 POI 列表
            recommended: 推薦的 POI 列表（已排序）
            k: Top-K
            
        Returns:
            Precision@K 分數
        """
        if k == 0 or len(recommended) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_k & relevant_set)
        return hits / k
    
    def recall_at_k(self, relevant: List, recommended: List, k: int) -> float:
        """
        計算 Recall@K
        
        Args:
            relevant: 實際相關的 POI 列表
            recommended: 推薦的 POI 列表（已排序）
            k: Top-K
            
        Returns:
            Recall@K 分數
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_k & relevant_set)
        return hits / len(relevant)
    
    def f1_score_at_k(self, relevant: List, recommended: List, k: int) -> float:
        """
        計算 F1-Score@K
        
        Args:
            relevant: 實際相關的 POI 列表
            recommended: 推薦的 POI 列表（已排序）
            k: Top-K
            
        Returns:
            F1-Score@K
        """
        precision = self.precision_at_k(relevant, recommended, k)
        recall = self.recall_at_k(relevant, recommended, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def dcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        計算 DCG@K (Discounted Cumulative Gain)
        
        Args:
            relevance_scores: 相關性分數列表（已排序）
            k: Top-K
            
        Returns:
            DCG@K 分數
        """
        k = min(k, len(relevance_scores))
        if k == 0:
            return 0.0
        
        dcg = relevance_scores[0]
        for i in range(1, k):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, relevant: List, recommended: List, scores: List[float], k: int) -> float:
        """
        計算 NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            relevant: 實際相關的 POI 列表
            recommended: 推薦的 POI 列表（已排序）
            scores: 推薦分數列表（與 recommended 對應）
            k: Top-K
            
        Returns:
            NDCG@K 分數
        """
        if len(relevant) == 0:
            return 0.0
        
        # 計算實際 DCG
        relevant_set = set(relevant)
        relevance_scores = []
        for i, poi in enumerate(recommended[:k]):
            if poi in relevant_set:
                relevance_scores.append(1.0)  # 二元相關性
            else:
                relevance_scores.append(0.0)
        
        dcg = self.dcg_at_k(relevance_scores, k)
        
        # 計算理想 DCG (IDCG)
        ideal_relevance = [1.0] * min(len(relevant), k)
        idcg = self.dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_auc(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """
        計算 AUC (Area Under ROC Curve)
        
        Args:
            labels: 真實標籤 (0 或 1)
            scores: 預測分數
            
        Returns:
            AUC 分數
        """
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            return roc_auc_score(labels, scores)
        except:
            return 0.0
    
    def get_user_recommendations(
        self, 
        user_id: str, 
        user_features: Dict[str, torch.Tensor],
        candidate_pois: List[str],
        top_k: int = 100
    ) -> Tuple[List[str], List[float]]:
        """
        為單個用戶生成推薦列表
        
        Args:
            user_id: 用戶 ID
            user_features: 用戶特徵
            candidate_pois: 候選 POI 列表
            top_k: 返回 Top-K 推薦
            
        Returns:
            (推薦 POI 列表, 推薦分數列表)
        """
        self.model.eval()
        
        batch_size = 512
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(candidate_pois), batch_size):
                batch_pois = candidate_pois[i:i+batch_size]
                batch_features = self._prepare_batch_features(
                    user_features, batch_pois
                )
                
                # 模型預測
                output = self.model(
                    batch_features['user_continuous'],
                    batch_features['user_categorical'],
                    batch_features['poi_continuous'],
                    batch_features['poi_categorical'],
                    batch_features['path_continuous']
                )
                
                # 提取評分（模型返回字典）
                scores = output['scores'] if isinstance(output, dict) else output
                all_scores.extend(scores.cpu().numpy().flatten().tolist())
        
        # 排序並返回 Top-K
        poi_scores = list(zip(candidate_pois, all_scores))
        poi_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_pois = [poi for poi, _ in poi_scores[:top_k]]
        top_scores = [score for _, score in poi_scores[:top_k]]
        
        return top_pois, top_scores
    
    def _prepare_batch_features(
        self, 
        user_features: Dict[str, torch.Tensor],
        poi_ids: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        準備批次特徵用於模型輸入
        
        Args:
            user_features: 用戶特徵（單個用戶）
            poi_ids: POI ID 列表
            
        Returns:
            批次特徵字典
        """
        batch_size = len(poi_ids)
        
        # 複製用戶特徵到批次大小
        batch_features = {
            'user_continuous': user_features['user_continuous'].repeat(batch_size, 1).to(self.device),
            'user_categorical': {},
            'poi_continuous': [],
            'poi_categorical': {},
            'path_continuous': torch.zeros(batch_size, 4).to(self.device)  # 評估時路徑特徵設為 0
        }
        
        # 複製用戶類別特徵
        for key, val in user_features['user_categorical'].items():
            batch_features['user_categorical'][key] = val.repeat(batch_size).to(self.device)
        
        # 獲取 POI 特徵
        poi_continuous_list = []
        poi_categorical_lists = {'category': [], 'state': [], 'price_level': []}
        
        for poi_id in poi_ids:
            # poi_index 儲存的是索引號，不是 POI 資料本身
            poi_idx = self.poi_processor.poi_index.get(poi_id)
            if poi_idx is None or poi_idx >= len(self.poi_processor.processed_pois):
                # 使用預設值
                poi_continuous = torch.zeros(8)
                poi_categorical_lists['category'].append(0)
                poi_categorical_lists['state'].append(0)
                poi_categorical_lists['price_level'].append(2)
            else:
                # 從 processed_pois 列表中獲取 POI 資料
                poi_data = self.poi_processor.processed_pois[poi_idx]
                
                poi_continuous = torch.tensor([
                    poi_data.get('avg_rating', 3.5),
                    poi_data.get('num_reviews', 0),  # 修正欄位名稱
                    poi_data.get('price_level', 2),
                    poi_data.get('latitude', 0),
                    poi_data.get('longitude', 0),
                    0,  # popularity_score (如果需要可計算)
                    0,  # 預留特徵
                    0   # distance (評估時未知)
                ], dtype=torch.float32)
                
                # 編碼類別特徵
                category_encoded = self.poi_processor.category_encoder.get(
                    poi_data.get('primary_category', 'Other'), 
                    self.poi_processor.category_encoder.get('Other', 0)
                )
                state_encoded = self.poi_processor.state_encoder.get(
                    poi_data.get('state', 'Unknown'),
                    self.poi_processor.state_encoder.get('Unknown', 0)
                )
                
                poi_categorical_lists['category'].append(category_encoded)
                poi_categorical_lists['state'].append(state_encoded)
                poi_categorical_lists['price_level'].append(min(poi_data.get('price_level', 2), 4))
            
            poi_continuous_list.append(poi_continuous)
        
        # 轉換為張量
        batch_features['poi_continuous'] = torch.stack(poi_continuous_list).to(self.device)
        
        # 組織 POI 類別特徵（與訓練時一致）
        for key in ['category', 'state', 'price_level']:
            batch_features['poi_categorical'][key] = torch.tensor(
                poi_categorical_lists[key], dtype=torch.long
            ).to(self.device)
        
        return batch_features
    
    def evaluate_user(
        self, 
        user_id: str,
        user_features: Dict[str, torch.Tensor],
        test_pois: List[str],
        all_candidate_pois: List[str]
    ) -> Dict[str, float]:
        """
        評估單個用戶的推薦性能
        
        Args:
            user_id: 用戶 ID
            user_features: 用戶特徵
            test_pois: 測試集中的正樣本 POI
            all_candidate_pois: 所有候選 POI
            
        Returns:
            評估指標字典
        """
        # 生成推薦列表
        max_k = max(self.k_values)
        recommended_pois, recommended_scores = self.get_user_recommendations(
            user_id, user_features, all_candidate_pois, top_k=max_k
        )
        
        # 調試：檢查推薦結果
        if len(recommended_pois) == 0:
            print(f"[警告] 用戶 {user_id} 沒有生成任何推薦")
            return {f'{metric}@{k}': 0.0 
                   for metric in ['precision', 'recall', 'f1', 'ndcg'] 
                   for k in self.k_values} | {'auc': 0.0}
        
        metrics = {}
        
        # 計算各個 K 值的指標
        for k in self.k_values:
            precision = self.precision_at_k(test_pois, recommended_pois, k)
            recall = self.recall_at_k(test_pois, recommended_pois, k)
            f1 = self.f1_score_at_k(test_pois, recommended_pois, k)
            ndcg = self.ndcg_at_k(test_pois, recommended_pois, recommended_scores, k)
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
            metrics[f'ndcg@{k}'] = ndcg
        
        # 計算 AUC（使用所有候選 POI）
        labels = []
        scores = []
        for poi, score in zip(recommended_pois, recommended_scores):
            labels.append(1 if poi in test_pois else 0)
            scores.append(score)
        
        if len(labels) > 0 and len(set(labels)) > 1:
            metrics['auc'] = self.calculate_auc(np.array(labels), np.array(scores))
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def evaluate_dataset(
        self,
        test_data: Dict[str, List[str]],
        user_features_dict: Dict[str, Dict[str, torch.Tensor]],
        all_candidate_pois: List[str],
        max_users: Optional[int] = None
    ) -> Dict[str, float]:
        """
        評估整個測試集
        
        Args:
            test_data: {user_id: [test_poi_ids]}
            user_features_dict: {user_id: user_features}
            all_candidate_pois: 所有候選 POI 列表
            max_users: 最多評估的用戶數（None 表示全部）
            
        Returns:
            平均評估指標
        """
        print(f"\n開始評估模型...")
        print(f"評估用戶數: {len(test_data)}")
        print(f"K 值: {self.k_values}")
        print(f"候選 POI 數量: {len(all_candidate_pois)}")
        
        all_metrics = defaultdict(list)
        
        users = list(test_data.keys())
        if max_users:
            users = users[:max_users]
        
        evaluated_count = 0
        skipped_no_features = 0
        skipped_no_test = 0
        error_count = 0
        
        for user_id in tqdm(users, desc="評估進度"):
            if user_id not in user_features_dict:
                skipped_no_features += 1
                continue
            
            test_pois = test_data[user_id]
            if len(test_pois) == 0:
                skipped_no_test += 1
                continue
            
            user_features = user_features_dict[user_id]
            
            try:
                metrics = self.evaluate_user(
                    user_id, user_features, test_pois, all_candidate_pois
                )
                
                for key, val in metrics.items():
                    all_metrics[key].append(val)
                
                evaluated_count += 1
                
                # 顯示前幾個用戶的詳細結果
                if evaluated_count <= 3:
                    print(f"\n[調試] 用戶 {user_id}:")
                    print(f"  測試 POI 數: {len(test_pois)}")
                    print(f"  Precision@1: {metrics.get('precision@1', 0):.4f}")
                    print(f"  Recall@1: {metrics.get('recall@1', 0):.4f}")
                    print(f"  NDCG@1: {metrics.get('ndcg@1', 0):.4f}")
                
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    print(f"\n評估用戶 {user_id} 時出錯: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        print(f"\n評估統計:")
        print(f"  成功評估: {evaluated_count} 個用戶")
        print(f"  跳過（無特徵）: {skipped_no_features} 個用戶")
        print(f"  跳過（無測試數據）: {skipped_no_test} 個用戶")
        print(f"  錯誤: {error_count} 個用戶")
        
        # 計算平均指標
        avg_metrics = {}
        for key, values in all_metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics


def load_model_and_processors(
    model_path: str,
    poi_data_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.nn.Module, POIDataProcessor]:
    """
    載入訓練好的模型和資料處理器
    
    Args:
        model_path: 模型檔案路徑
        poi_data_path: POI 資料路徑
        device: 計算設備
        
    Returns:
        (模型, POI 處理器)
    """
    print(f"\n載入模型和資料...")
    print(f"模型路徑: {model_path}")
    print(f"設備: {device}")
    
    # 載入 POI 資料
    poi_processor = POIDataProcessor(poi_data_path)
    poi_processor.load_data()
    print(f"✓ 載入了 {len(poi_processor.pois)} 個 POI")
    if len(poi_processor.pois) > 0:
        sample_poi = poi_processor.pois[0]
        print(f"  POI 欄位: {list(sample_poi.keys())[:10]}")
    
    poi_processor.preprocess()
    print(f"✓ 處理後有 {len(poi_processor.processed_pois)} 個 POI")
    if len(poi_processor.processed_pois) > 0:
        sample_processed = poi_processor.processed_pois[0]
        print(f"  處理後欄位: {list(sample_processed.keys())[:10]}")
    
    # 載入 checkpoint 以獲取模型配置
    checkpoint = torch.load(model_path, map_location=device)
    
    # 從 checkpoint 中獲取 vocab_sizes（如果有）
    if 'poi_vocab_sizes' in checkpoint:
        poi_vocab_sizes = checkpoint['poi_vocab_sizes']
        print(f"  從 checkpoint 載入 vocab_sizes: {poi_vocab_sizes}")
    else:
        # 使用訓練時的配置（與 train_model.py 一致）
        poi_vocab_sizes = {
            'category': len(poi_processor.category_encoder),
            'state': len(poi_processor.state_encoder),
            'price_level': 5
        }
        print(f"  使用預設 vocab_sizes: {poi_vocab_sizes}")
    
    # 創建模型
    user_vocab_sizes = {}  # 訓練時沒有使用用戶類別特徵
    
    model = create_travel_dlrm(
        user_continuous_dim=10,
        poi_continuous_dim=8,
        path_continuous_dim=4,
        user_vocab_sizes=user_vocab_sizes,
        poi_vocab_sizes=poi_vocab_sizes,
        embedding_dim=64,
        bottom_mlp_dims=[256, 128],
        top_mlp_dims=[512, 256, 128],
        dropout=0.2
    )
    
    # 載入訓練好的權重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("✓ 模型載入完成!")
    
    return model, poi_processor


def prepare_test_data(
    review_data_path: str,
    train_ratio: float = 0.8
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    準備訓練集和測試集
    
    Args:
        review_data_path: 評論資料路徑
        train_ratio: 訓練集比例
        
    Returns:
        (訓練集, 測試集) - {user_id: [poi_ids]}
    """
    print(f"\n準備測試資料...")
    print(f"評論資料路徑: {review_data_path}")
    
    review_processor = ReviewDataProcessor(review_data_path)
    review_processor.load_data(max_records=50000)  # 載入部分資料進行評估
    
    print(f"✓ 載入了 {len(review_processor.reviews)} 條評論")
    
    # 檢查資料格式
    if len(review_processor.reviews) > 0:
        sample = review_processor.reviews[0]
        print(f"  評論欄位: {list(sample.keys())[:10]}")
    
    user_interactions = defaultdict(list)
    
    for review in review_processor.reviews:
        user_id = review.get('user_id')
        # 嘗試不同的 POI ID 鍵名
        poi_id = review.get('gmap_id') or review.get('business_id')
        rating = review.get('stars', 0) or review.get('rating', 0)
        
        if not user_id or not poi_id:
            continue
        
        # 只考慮高評分（4-5 星）作為正樣本
        if rating >= 4.0:
            user_interactions[user_id].append(poi_id)
    
    print(f"✓ 找到 {len(user_interactions)} 個用戶的互動記錄")
    
    # 分割訓練集和測試集
    train_data = {}
    test_data = {}
    
    for user_id, pois in user_interactions.items():
        if len(pois) < 3:  # 至少需要 3 個互動
            continue
        
        n_train = max(1, int(len(pois) * train_ratio))
        train_data[user_id] = pois[:n_train]
        test_data[user_id] = pois[n_train:]
    
    print(f"✓ 訓練集用戶數: {len(train_data)}")
    print(f"✓ 測試集用戶數: {len(test_data)}")
    
    return train_data, test_data


def prepare_user_features(
    train_data: Dict[str, List[str]],
    poi_processor: POIDataProcessor,
    device: str = 'cpu'
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    根據訓練資料準備用戶特徵
    
    Args:
        train_data: 訓練集 {user_id: [poi_ids]}
        poi_processor: POI 處理器
        device: 計算設備
        
    Returns:
        {user_id: user_features}
    """
    print(f"\n準備用戶特徵...")
    
    user_features_dict = {}
    
    for user_id, poi_ids in tqdm(train_data.items(), desc="處理用戶"):
        # 計算用戶統計特徵
        ratings = []
        categories = []
        
        for poi_id in poi_ids:
            # poi_index 儲存的是索引號，需要從 processed_pois 列表獲取資料
            poi_idx = poi_processor.poi_index.get(poi_id)
            if poi_idx is not None and poi_idx < len(poi_processor.processed_pois):
                poi_data = poi_processor.processed_pois[poi_idx]
                ratings.append(poi_data.get('avg_rating', 3.5))
                categories.append(poi_data.get('primary_category', ''))
        
        if len(ratings) == 0:
            continue
        
        # 用戶連續特徵
        avg_rating = np.mean(ratings)
        std_rating = np.std(ratings) if len(ratings) > 1 else 0.5
        num_reviews = len(poi_ids)
        
        user_continuous = torch.tensor([
            avg_rating,
            std_rating,
            num_reviews,
            0, 0, 0, 0, 0, 0, 0  # 填充到 10 維
        ], dtype=torch.float32)
        
        # 用戶類別特徵（訓練時為空字典）
        user_categorical = {}
        
        user_features_dict[user_id] = {
            'user_continuous': user_continuous,
            'user_categorical': user_categorical
        }
    
    print(f"✓ 完成 {len(user_features_dict)} 個用戶特徵準備")
    
    return user_features_dict


def main():
    parser = argparse.ArgumentParser(description='評估旅遊推薦模型')
    parser.add_argument('--model', type=str, default='models/travel_dlrm.pth',
                        help='模型檔案路徑')
    parser.add_argument('--poi-data', type=str, default='datasets/meta-California.json.gz',
                        help='POI 資料路徑')
    parser.add_argument('--review-data', type=str, default='datasets/review-California.json.gz',
                        help='評論資料路徑')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 3, 10],
                        help='K 值列表')
    parser.add_argument('--max-users', type=int, default=500,
                        help='最多評估的用戶數')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='計算設備')
    
    args = parser.parse_args()
    
    print("="*60)
    print("旅遊推薦系統 - 模型評估")
    print("="*60)
    
    # 載入模型和資料
    model, poi_processor = load_model_and_processors(
        args.model, args.poi_data, args.device
    )
    
    # 準備測試資料
    train_data, test_data = prepare_test_data(args.review_data)
    
    # 準備用戶特徵
    user_features_dict = prepare_user_features(train_data, poi_processor, args.device)
    
    # 獲取所有候選 POI
    # 注意：processed_pois 使用 'id' 欄位（來自原始的 'gmap_id'）
    all_candidate_pois = [poi['id'] for poi in poi_processor.processed_pois 
                          if 'id' in poi]
    
    print(f"\n✓ 候選 POI 總數: {len(all_candidate_pois)}")
    if len(poi_processor.processed_pois) > 0 and len(all_candidate_pois) == 0:
        print("⚠️  警告：POI 資料中沒有 'id' 欄位！")
        sample_keys = list(poi_processor.processed_pois[0].keys())
        print(f"   可用欄位: {sample_keys}")
    
    # ================= 插入此段 Debug 程式碼 =================
    print("\n" + "="*20 + " DEBUG START " + "="*20)
    
    # 檢查 1: 候選 POI 是否為空？
    print(f"候選 POI 數量: {len(all_candidate_pois)}")
    if len(all_candidate_pois) > 0:
        print(f"候選 POI 範例 (前3): {all_candidate_pois[:3]}")
    else:
        print("❌ 嚴重錯誤: all_candidate_pois 為空！請檢查 poi['id'] 鍵值名稱。")

    # 檢查 2: 測試集 ID 範例
    test_user_sample = list(test_data.keys())[0]
    test_poi_sample = test_data[test_user_sample]
    print(f"測試集 User ID: {test_user_sample}")
    print(f"測試集 POI ID 範例: {test_poi_sample[:3]}")

    # 檢查 3: ID 是否有交集？
    candidate_set = set(all_candidate_pois)
    test_poi_set = set(test_poi_sample)
    intersection = candidate_set.intersection(test_poi_set)
    
    print(f"單一用戶測試集與候選集的交集數量: {len(intersection)}")
    if len(intersection) == 0:
        print("❌ 嚴重錯誤: ID 完全對不上！Review 中的 ID 不存在於 POI 列表中。")
        print("可能原因：")
        print("1. Review 用的是 'gmap_id' 但 POI 用的是 'business_id'")
        print("2. 資料處理時 ID 被去除了空白或改變了格式")
        print("3. poi_processor.processed_pois 裡面的 key 不是 'id' 而是 'gmap_id'")
    else:
        print("✅ ID 格式看起來正常，有交集。")
        
    # 檢查 4: 處理器索引
    print(f"Processor Index Size: {len(poi_processor.poi_index)}")
    sample_lookup_id = test_poi_sample[0]
    lookup_result = poi_processor.poi_index.get(sample_lookup_id)
    print(f"嘗試從 poi_index 查找 ID '{sample_lookup_id}': 索引值 = {lookup_result}")
    
    print("="*20 + " DEBUG END " + "="*20 + "\n")
    # =======================================================
    
    # 創建評估器
    evaluator = RecommenderEvaluator(model, poi_processor, k_values=args.k_values)
    
    # 執行評估
    avg_metrics = evaluator.evaluate_dataset(
        test_data,
        user_features_dict,
        all_candidate_pois,
        max_users=args.max_users
    )
    
    # 顯示結果
    print("\n" + "="*60)
    print("評估結果")
    print("="*60)
    
    for k in args.k_values:
        print(f"\n--- K = {k} ---")
        print(f"Precision@{k}: {avg_metrics.get(f'precision@{k}', 0):.4f}")
        print(f"Recall@{k}:    {avg_metrics.get(f'recall@{k}', 0):.4f}")
        print(f"F1-Score@{k}:  {avg_metrics.get(f'f1@{k}', 0):.4f}")
        print(f"NDCG@{k}:      {avg_metrics.get(f'ndcg@{k}', 0):.4f}")
    
    print(f"\n--- 整體指標 ---")
    print(f"AUC: {avg_metrics.get('auc', 0):.4f}")
    
    # 儲存結果
    output_file = 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(avg_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 評估結果已儲存至: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
