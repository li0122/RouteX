"""
評估 DLRM 模型的推薦效果
計算 NDCG@k, Recall@k, Precision@k, MRR, MAP 等指標
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict

from dlrm_model import create_travel_dlrm
from route_aware_recommender import create_route_aware_recommender


class RecommendationMetrics:
    """推薦系統評估指標"""
    
    @staticmethod
    def ndcg_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """
        計算 NDCG@k (Normalized Discounted Cumulative Gain)
        
        Args:
            predictions: 預測的 POI ID 列表（按推薦順序）
            ground_truth: 真實互動的 POI ID 列表
            k: 截斷位置
        
        Returns:
            NDCG@k 分數 [0, 1]
        """
        if not ground_truth:
            return 0.0
        
        # DCG@k
        dcg = 0.0
        for i, pred_id in enumerate(predictions[:k]):
            if pred_id in ground_truth:
                # rel = 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
        
        # IDCG@k (理想情況：前k個都是相關的)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def recall_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """
        計算 Recall@k
        
        Args:
            predictions: 預測的 POI ID 列表
            ground_truth: 真實互動的 POI ID 列表
            k: 截斷位置
        
        Returns:
            Recall@k 分數 [0, 1]
        """
        if not ground_truth:
            return 0.0
        
        top_k_preds = set(predictions[:k])
        relevant = set(ground_truth)
        
        hits = len(top_k_preds & relevant)
        return hits / len(relevant)
    
    @staticmethod
    def precision_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """
        計算 Precision@k
        
        Args:
            predictions: 預測的 POI ID 列表
            ground_truth: 真實互動的 POI ID 列表
            k: 截斷位置
        
        Returns:
            Precision@k 分數 [0, 1]
        """
        if k == 0:
            return 0.0
        
        top_k_preds = set(predictions[:k])
        relevant = set(ground_truth)
        
        hits = len(top_k_preds & relevant)
        return hits / k
    
    @staticmethod
    def hit_rate_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """
        計算 Hit Rate@k (至少命中一個)
        
        Args:
            predictions: 預測的 POI ID 列表
            ground_truth: 真實互動的 POI ID 列表
            k: 截斷位置
        
        Returns:
            Hit Rate@k: 1.0 if hit, 0.0 otherwise
        """
        top_k_preds = set(predictions[:k])
        relevant = set(ground_truth)
        
        return 1.0 if len(top_k_preds & relevant) > 0 else 0.0
    
    @staticmethod
    def mrr(predictions: List[int], ground_truth: List[int]) -> float:
        """
        計算 MRR (Mean Reciprocal Rank)
        
        Args:
            predictions: 預測的 POI ID 列表
            ground_truth: 真實互動的 POI ID 列表
        
        Returns:
            MRR 分數
        """
        relevant = set(ground_truth)
        
        for i, pred_id in enumerate(predictions):
            if pred_id in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def average_precision(predictions: List[int], ground_truth: List[int]) -> float:
        """
        計算 Average Precision (AP)
        
        Args:
            predictions: 預測的 POI ID 列表
            ground_truth: 真實互動的 POI ID 列表
        
        Returns:
            AP 分數
        """
        if not ground_truth:
            return 0.0
        
        relevant = set(ground_truth)
        hits = 0
        sum_precisions = 0.0
        
        for i, pred_id in enumerate(predictions):
            if pred_id in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        if hits == 0:
            return 0.0
        
        return sum_precisions / len(relevant)


class ModelEvaluator:
    """模型評估器"""
    
    def __init__(
        self,
        model_path: str = "models/travel_dlrm.pth",
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.recommender = None
        self.metrics = RecommendationMetrics()
        
    def load_model(self):
        """載入訓練好的模型"""
        print(f"📦 載入模型: {self.model_path}")
        
        try:
            # 載入 checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 從 checkpoint 獲取模型配置（如果有）
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"   配置: {config}")
            else:
                # 使用預設配置
                config = {
                    'user_continuous_dim': 10,
                    'poi_continuous_dim': 8,
                    'path_continuous_dim': 4,
                    'poi_vocab_sizes': {
                        'category': 100,
                        'state': 50,
                        'price_level': 5
                    },
                    'user_vocab_sizes': {},
                    'embedding_dim': 64
                }
                print(f"   使用預設配置")
            
            # 創建模型
            self.model = create_travel_dlrm(**config)
            
            # 載入權重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ 模型權重載入成功")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✓ 模型權重載入成功")
            
            self.model.to(self.device)
            self.model.eval()
            
            # 顯示模型資訊
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"   參數量: {num_params:,}")
            
            if 'epoch' in checkpoint:
                print(f"   訓練輪次: {checkpoint['epoch']}")
            if 'best_loss' in checkpoint:
                print(f"   最佳損失: {checkpoint['best_loss']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_test_data(self, test_data_path: Optional[str] = None) -> List[Dict]:
        """
        載入測試數據
        
        Args:
            test_data_path: 測試數據路徑（JSON格式）
        
        Returns:
            測試樣本列表
        """
        if test_data_path and Path(test_data_path).exists():
            print(f"📂 載入測試數據: {test_data_path}")
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"✓ 載入 {len(test_data)} 個測試樣本")
            return test_data
        else:
            print("⚠️ 未提供測試數據，生成模擬數據...")
            return self._generate_mock_test_data()
    
    def _generate_mock_test_data(self, num_users: int = 50) -> List[Dict]:
        """生成模擬測試數據"""
        test_data = []
        
        # 模擬用戶和POI
        for user_id in range(num_users):
            sample = {
                'user_id': f"user_{user_id}",
                'user_history': [
                    {'poi_id': f"poi_{i}", 'rating': 4.0 + np.random.rand()}
                    for i in range(5)
                ],
                'start_location': (37.7749 + np.random.randn() * 0.01, 
                                  -122.4194 + np.random.randn() * 0.01),
                'end_location': (37.7849 + np.random.randn() * 0.01, 
                                -122.4094 + np.random.randn() * 0.01),
                'ground_truth': [f"poi_{i}" for i in range(100, 105)],  # 真實互動的POI
                'activity_intent': '旅遊探索'
            }
            test_data.append(sample)
        
        print(f"✓ 生成 {len(test_data)} 個模擬測試樣本")
        return test_data
    
    def evaluate(
        self,
        test_data: List[Dict],
        k_values: List[int] = [5, 10, 20],
        top_k: int = 50
    ) -> Dict:
        """
        評估模型
        
        Args:
            test_data: 測試數據列表
            k_values: 要計算的 k 值列表
            top_k: 生成的推薦數量
        
        Returns:
            評估結果字典
        """
        print(f"\n🎯 開始評估模型...")
        print(f"   測試樣本數: {len(test_data)}")
        print(f"   評估指標: NDCG@k, Recall@k, Precision@k, Hit Rate@k, MRR, MAP")
        print(f"   k 值: {k_values}")
        
        # 初始化結果容器
        results = {
            'ndcg': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'precision': {k: [] for k in k_values},
            'hit_rate': {k: [] for k in k_values},
            'mrr': [],
            'map': []
        }
        
        # 載入推薦器（如果需要）
        if self.recommender is None:
            print("\n📦 初始化推薦器...")
            try:
                self.recommender = create_route_aware_recommender(
                    model_checkpoint=self.model_path,
                    device=str(self.device)
                )
            except Exception as e:
                print(f"⚠️ 推薦器初始化失敗: {e}")
                print("⚠️ 將使用簡化評估模式（僅模型推理）")
        
        # 對每個測試樣本進行評估
        for idx, sample in enumerate(test_data):
            if (idx + 1) % 10 == 0:
                print(f"   進度: {idx + 1}/{len(test_data)}")
            
            try:
                # 獲取推薦結果
                if self.recommender:
                    recommendations = self.recommender.recommend_on_route(
                        user_id=sample['user_id'],
                        user_history=sample.get('user_history', []),
                        start_location=sample['start_location'],
                        end_location=sample['end_location'],
                        activityIntent=sample.get('activity_intent', '旅遊探索'),
                        top_k=top_k
                    )
                    
                    # 提取推薦的 POI ID
                    predicted_ids = [rec['poi'].get('poi_id', rec['poi'].get('name', f"poi_{i}")) 
                                    for i, rec in enumerate(recommendations)]
                else:
                    # 簡化模式：生成隨機預測
                    predicted_ids = [f"poi_{i}" for i in range(top_k)]
                
                ground_truth = sample.get('ground_truth', [])
                
                # 計算各項指標
                for k in k_values:
                    results['ndcg'][k].append(
                        self.metrics.ndcg_at_k(predicted_ids, ground_truth, k)
                    )
                    results['recall'][k].append(
                        self.metrics.recall_at_k(predicted_ids, ground_truth, k)
                    )
                    results['precision'][k].append(
                        self.metrics.precision_at_k(predicted_ids, ground_truth, k)
                    )
                    results['hit_rate'][k].append(
                        self.metrics.hit_rate_at_k(predicted_ids, ground_truth, k)
                    )
                
                results['mrr'].append(
                    self.metrics.mrr(predicted_ids, ground_truth)
                )
                results['map'].append(
                    self.metrics.average_precision(predicted_ids, ground_truth)
                )
                
            except Exception as e:
                print(f"⚠️ 樣本 {idx} 評估失敗: {e}")
                continue
        
        # 計算平均值
        avg_results = {
            'ndcg': {k: np.mean(v) if v else 0.0 for k, v in results['ndcg'].items()},
            'recall': {k: np.mean(v) if v else 0.0 for k, v in results['recall'].items()},
            'precision': {k: np.mean(v) if v else 0.0 for k, v in results['precision'].items()},
            'hit_rate': {k: np.mean(v) if v else 0.0 for k, v in results['hit_rate'].items()},
            'mrr': np.mean(results['mrr']) if results['mrr'] else 0.0,
            'map': np.mean(results['map']) if results['map'] else 0.0,
            'num_samples': len(test_data)
        }
        
        return avg_results
    
    def print_results(self, results: Dict):
        """打印評估結果"""
        print("\n" + "="*60)
        print("📊 評估結果")
        print("="*60)
        
        print(f"\n樣本數: {results['num_samples']}")
        
        print(f"\n📈 NDCG (Normalized Discounted Cumulative Gain):")
        for k, score in results['ndcg'].items():
            print(f"   NDCG@{k:2d}: {score:.4f}")
        
        print(f"\n📈 Recall:")
        for k, score in results['recall'].items():
            print(f"   Recall@{k:2d}: {score:.4f}")
        
        print(f"\n📈 Precision:")
        for k, score in results['precision'].items():
            print(f"   Precision@{k:2d}: {score:.4f}")
        
        print(f"\n📈 Hit Rate:")
        for k, score in results['hit_rate'].items():
            print(f"   Hit Rate@{k:2d}: {score:.4f}")
        
        print(f"\n📈 其他指標:")
        print(f"   MRR (Mean Reciprocal Rank): {results['mrr']:.4f}")
        print(f"   MAP (Mean Average Precision): {results['map']:.4f}")
        
        print("\n" + "="*60)
    
    def save_results(self, results: Dict, output_path: str = "evaluation_results.json"):
        """保存評估結果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 評估結果已保存至: {output_path}")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='評估 Travel DLRM 模型')
    parser.add_argument('--model', type=str, default='models/travel_dlrm.pth',
                       help='模型權重路徑')
    parser.add_argument('--test-data', type=str, default=None,
                       help='測試數據路徑 (JSON)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='運算設備')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                       help='評估的 k 值列表')
    parser.add_argument('--top-k', type=int, default=50,
                       help='生成的推薦數量')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='結果輸出路徑')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 Travel DLRM 模型評估")
    print("="*60)
    
    # 創建評估器
    evaluator = ModelEvaluator(
        model_path=args.model,
        device=args.device
    )
    
    # 載入模型
    if not evaluator.load_model():
        print("❌ 模型載入失敗，退出評估")
        return
    
    # 載入測試數據
    test_data = evaluator.load_test_data(args.test_data)
    
    # 執行評估
    results = evaluator.evaluate(
        test_data=test_data,
        k_values=args.k_values,
        top_k=args.top_k
    )
    
    # 打印結果
    evaluator.print_results(results)
    
    # 保存結果
    evaluator.save_results(results, args.output)
    
    print("\n✅ 評估完成!")


if __name__ == "__main__":
    main()
