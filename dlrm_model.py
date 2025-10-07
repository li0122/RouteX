"""
DLRM (Deep Learning Recommendation Model) 實現
用於旅行推薦系統
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class EmbeddingLayer(nn.Module):
    """嵌入層 - 處理類別特徵"""
    
    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, embedding_dim)
            for name, size in vocab_sizes.items()
        })
        self.embedding_dim = embedding_dim
    
    def forward(self, categorical_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            categorical_features: {feature_name: indices} 形狀 (batch_size,)
        Returns:
            embeddings: (batch_size, num_features, embedding_dim)
        """
        embedded = []
        for name, indices in categorical_features.items():
            if name in self.embeddings:
                emb = self.embeddings[name](indices)
                embedded.append(emb)
        
        if embedded:
            return torch.stack(embedded, dim=1)
        else:
            # 如果沒有類別特徵，返回 None
            return None


class BottomMLP(nn.Module):
    """底層MLP - 處理連續特徵"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.mlp(x)


class FeatureInteraction(nn.Module):
    """特徵交互層 - DLRM 核心"""
    
    def __init__(self, method: str = "dot"):
        super().__init__()
        self.method = method
    
    def forward(self, bottom_output: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bottom_output: (batch_size, embedding_dim) - 來自底層MLP
            embeddings: (batch_size, num_embeddings, embedding_dim) - 類別特徵嵌入
        Returns:
            interactions: (batch_size, interaction_dim)
        """
        batch_size = bottom_output.size(0)
        embedding_dim = bottom_output.size(1)
        num_embeddings = embeddings.size(1)
        
        # 將 bottom_output 視為一個額外的嵌入
        # (batch_size, 1, embedding_dim)
        bottom_emb = bottom_output.unsqueeze(1)
        
        # 合併所有嵌入: (batch_size, num_embeddings + 1, embedding_dim)
        all_embeddings = torch.cat([bottom_emb, embeddings], dim=1)
        
        # 計算所有配對的點積
        interactions = []
        num_features = all_embeddings.size(1)
        
        for i in range(num_features):
            for j in range(i + 1, num_features):
                # 點積交互
                interaction = torch.sum(
                    all_embeddings[:, i, :] * all_embeddings[:, j, :], 
                    dim=1, keepdim=True
                )
                interactions.append(interaction)
        
        if interactions:
            # (batch_size, num_interactions)
            return torch.cat(interactions, dim=1)
        else:
            return torch.zeros(batch_size, 1, device=bottom_output.device)


class TopMLP(nn.Module):
    """頂層MLP - 最終預測"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 輸出層 - 評分預測
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            scores: (batch_size, 1)
        """
        return self.mlp(x)


class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM)
    整合連續特徵、類別特徵和特徵交互
    """
    
    def __init__(
        self,
        continuous_dim: int,
        vocab_sizes: Dict[str, int],
        embedding_dim: int = 64,
        bottom_mlp_dims: List[int] = [256, 128],
        top_mlp_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 1. 嵌入層 - 類別特徵
        self.embedding_layer = EmbeddingLayer(vocab_sizes, embedding_dim)
        
        # 2. 底層MLP - 連續特徵
        self.bottom_mlp = BottomMLP(
            input_dim=continuous_dim,
            hidden_dims=bottom_mlp_dims,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # 3. 特徵交互層
        self.feature_interaction = FeatureInteraction(method="dot")
        
        # 4. 計算交互數量
        num_embeddings = len(vocab_sizes) + 1  # +1 for bottom_mlp output
        num_interactions = num_embeddings * (num_embeddings - 1) // 2
        
        # 5. 頂層MLP
        # 輸入 = 交互向量 + 原始嵌入向量
        top_mlp_input_dim = num_interactions + num_embeddings * embedding_dim
        
        self.top_mlp = TopMLP(
            input_dim=top_mlp_input_dim,
            hidden_dims=top_mlp_dims,
            dropout=dropout
        )
    
    def forward(
        self, 
        continuous_features: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            continuous_features: (batch_size, continuous_dim)
            categorical_features: {feature_name: (batch_size,)}
        
        Returns:
            scores: (batch_size, 1)
        """
        # 1. 處理連續特徵
        bottom_output = self.bottom_mlp(continuous_features)  # (batch_size, embedding_dim)
        
        # 2. 處理類別特徵
        embeddings = self.embedding_layer(categorical_features)  # (batch_size, num_embeddings, embedding_dim)
        
        # 3. 特徵交互
        interactions = self.feature_interaction(bottom_output, embeddings)  # (batch_size, num_interactions)
        
        # 4. 合併特徵
        # 展平嵌入
        batch_size = bottom_output.size(0)
        bottom_emb = bottom_output.unsqueeze(1)
        all_embeddings = torch.cat([bottom_emb, embeddings], dim=1)  # (batch_size, num_embeddings+1, embedding_dim)
        flat_embeddings = all_embeddings.view(batch_size, -1)  # (batch_size, (num_embeddings+1) * embedding_dim)
        
        # 合併交互和原始嵌入
        top_input = torch.cat([interactions, flat_embeddings], dim=1)
        
        # 5. 最終預測
        scores = self.top_mlp(top_input)
        
        return scores


class TravelDLRM(nn.Module):
    """
    旅行推薦專用的 DLRM 模型
    整合用戶偏好、POI特徵和路徑資訊
    """
    
    def __init__(
        self,
        user_continuous_dim: int,
        poi_continuous_dim: int,
        path_continuous_dim: int,
        user_vocab_sizes: Dict[str, int],
        poi_vocab_sizes: Dict[str, int],
        embedding_dim: int = 64,
        bottom_mlp_dims: List[int] = [256, 128],
        top_mlp_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 用戶特徵處理
        self.user_embeddings = EmbeddingLayer(user_vocab_sizes, embedding_dim)
        self.user_mlp = BottomMLP(
            user_continuous_dim, bottom_mlp_dims, embedding_dim, dropout
        )
        
        # POI特徵處理
        self.poi_embeddings = EmbeddingLayer(poi_vocab_sizes, embedding_dim)
        self.poi_mlp = BottomMLP(
            poi_continuous_dim, bottom_mlp_dims, embedding_dim, dropout
        )
        
        # 路徑特徵處理
        self.path_mlp = BottomMLP(
            path_continuous_dim, [64, 32], embedding_dim, dropout
        )
        
        # 特徵交互
        self.feature_interaction = FeatureInteraction(method="dot")
        
        # 計算頂層MLP輸入維度
        # bottom MLPs 總是有 3 個 (user, poi, path)
        num_embeddings = 3 + len(user_vocab_sizes) + len(poi_vocab_sizes)
        # feature_interaction 會將 bottom_output 也加入交互，所以是 num_embeddings + 1
        num_for_interaction = num_embeddings + 1
        num_interactions = num_for_interaction * (num_for_interaction - 1) // 2
        top_mlp_input_dim = num_interactions + num_embeddings * embedding_dim
        
        # 頂層預測
        self.top_mlp = TopMLP(top_mlp_input_dim, top_mlp_dims, dropout)
        
        # 注意力機制 - 用於路徑感知
        self.path_attention = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        user_continuous: torch.Tensor,
        user_categorical: Dict[str, torch.Tensor],
        poi_continuous: torch.Tensor,
        poi_categorical: Dict[str, torch.Tensor],
        path_continuous: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            user_continuous: (batch_size, user_continuous_dim)
            user_categorical: {feature_name: (batch_size,)}
            poi_continuous: (batch_size, poi_continuous_dim)
            poi_categorical: {feature_name: (batch_size,)}
            path_continuous: (batch_size, path_continuous_dim)
        
        Returns:
            {
                'scores': (batch_size, 1),
                'attention_weights': (batch_size, num_embeddings, 1)
            }
        """
        batch_size = user_continuous.size(0)
        
        # 1. 處理用戶特徵
        user_bottom = self.user_mlp(user_continuous)  # (batch_size, embedding_dim)
        user_embs = self.user_embeddings(user_categorical)  # (batch_size, num_user_embs, embedding_dim) or None
        
        # 2. 處理POI特徵
        poi_bottom = self.poi_mlp(poi_continuous)
        poi_embs = self.poi_embeddings(poi_categorical)
        
        # 3. 處理路徑特徵
        path_bottom = self.path_mlp(path_continuous)
        
        # 4. 合併所有嵌入
        bottom_embs = torch.stack([user_bottom, poi_bottom, path_bottom], dim=1)  # (batch_size, 3, embedding_dim)
        
        # 只合併非 None 的 embeddings
        embeddings_list = [bottom_embs]
        if user_embs is not None:
            embeddings_list.append(user_embs)
        if poi_embs is not None:
            embeddings_list.append(poi_embs)
        
        all_embeddings = torch.cat(embeddings_list, dim=1)  # (batch_size, num_total_embs, embedding_dim)
        
        # 5. 路徑注意力
        attention_weights = self.path_attention(all_embeddings)  # (batch_size, num_total_embs, 1)
        weighted_embeddings = all_embeddings * attention_weights
        
        # 6. 特徵交互
        # 使用加權平均作為bottom_output
        bottom_for_interaction = weighted_embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        interactions = self.feature_interaction(
            bottom_for_interaction, 
            all_embeddings
        )
        
        # 7. 合併特徵
        flat_embeddings = all_embeddings.view(batch_size, -1)
        top_input = torch.cat([interactions, flat_embeddings], dim=1)
        
        # 8. 最終評分
        scores = self.top_mlp(top_input)
        
        return {
            'scores': scores,
            'attention_weights': attention_weights,
            'user_embedding': user_bottom,
            'poi_embedding': poi_bottom,
            'path_embedding': path_bottom
        }
    
    def predict(
        self,
        user_continuous: torch.Tensor,
        user_categorical: Dict[str, torch.Tensor],
        poi_continuous: torch.Tensor,
        poi_categorical: Dict[str, torch.Tensor],
        path_continuous: torch.Tensor
    ) -> torch.Tensor:
        """
        預測評分（推理模式）
        
        Returns:
            scores: (batch_size,)
        """
        with torch.no_grad():
            output = self.forward(
                user_continuous, user_categorical,
                poi_continuous, poi_categorical,
                path_continuous
            )
            return output['scores'].squeeze(-1)


class DLRMLoss(nn.Module):
    """DLRM 損失函數"""
    
    def __init__(self, loss_type: str = "bce", margin: float = 0.5):
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin
    
    def forward(
        self, 
        pos_scores: torch.Tensor, 
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        計算損失
        
        Args:
            pos_scores: (batch_size, 1) - 正樣本分數
            neg_scores: (batch_size, 1) - 負樣本分數
        
        Returns:
            loss: scalar
        """
        if self.loss_type == "bce":
            # Binary Cross Entropy
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            return (pos_loss + neg_loss) / 2
        
        elif self.loss_type == "bpr":
            # Bayesian Personalized Ranking
            diff = pos_scores - neg_scores
            return -F.logsigmoid(diff).mean()
        
        elif self.loss_type == "margin":
            # Margin Ranking Loss
            return F.relu(self.margin - (pos_scores - neg_scores)).mean()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def create_travel_dlrm(
    user_continuous_dim: int = 10,
    poi_continuous_dim: int = 8,
    path_continuous_dim: int = 4,
    user_vocab_sizes: Optional[Dict[str, int]] = None,
    poi_vocab_sizes: Optional[Dict[str, int]] = None,
    embedding_dim: int = 64,
    **kwargs
) -> TravelDLRM:
    """
    創建旅行推薦 DLRM 模型的工廠函數
    
    Args:
        user_continuous_dim: 用戶連續特徵維度
        poi_continuous_dim: POI連續特徵維度
        path_continuous_dim: 路徑連續特徵維度
        user_vocab_sizes: 用戶類別特徵詞彙表大小
        poi_vocab_sizes: POI類別特徵詞彙表大小
        embedding_dim: 嵌入維度
    
    Returns:
        TravelDLRM 模型實例
    """
    if user_vocab_sizes is None:
        user_vocab_sizes = {}
    
    if poi_vocab_sizes is None:
        poi_vocab_sizes = {}
    
    model = TravelDLRM(
        user_continuous_dim=user_continuous_dim,
        poi_continuous_dim=poi_continuous_dim,
        path_continuous_dim=path_continuous_dim,
        user_vocab_sizes=user_vocab_sizes,
        poi_vocab_sizes=poi_vocab_sizes,
        embedding_dim=embedding_dim,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 測試模型
    print("=== DLRM 模型測試 ===")
    
    # 定義詞彙表大小
    user_vocab_sizes = {
        'age_group': 10,
        'gender': 3,
        'travel_style': 20
    }
    
    poi_vocab_sizes = {
        'category': 100,
        'price_level': 5,
        'city': 50
    }
    
    # 創建模型
    model = create_travel_dlrm(
        user_continuous_dim=10,
        poi_continuous_dim=8,
        path_continuous_dim=4,
        user_vocab_sizes=user_vocab_sizes,
        poi_vocab_sizes=poi_vocab_sizes,
        embedding_dim=32
    )
    
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    batch_size = 4
    
    user_continuous = torch.randn(batch_size, 10)
    user_categorical = {
        'age_group': torch.randint(0, 10, (batch_size,)),
        'gender': torch.randint(0, 3, (batch_size,)),
        'travel_style': torch.randint(0, 20, (batch_size,))
    }
    
    poi_continuous = torch.randn(batch_size, 8)
    poi_categorical = {
        'category': torch.randint(0, 100, (batch_size,)),
        'price_level': torch.randint(0, 5, (batch_size,)),
        'city': torch.randint(0, 50, (batch_size,))
    }
    
    path_continuous = torch.randn(batch_size, 4)
    
    # 前向傳播
    output = model(
        user_continuous, user_categorical,
        poi_continuous, poi_categorical,
        path_continuous
    )
    
    print(f"\n輸出形狀:")
    print(f"  scores: {output['scores'].shape}")
    print(f"  attention_weights: {output['attention_weights'].shape}")
    
    print(f"\n預測分數:")
    print(output['scores'].squeeze())
    
    print("\n✓ DLRM 模型測試成功!")
