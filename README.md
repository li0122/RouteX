# 旅行推薦系統

基於 **DLRM (Deep Learning Recommendation Model)** 和 **OSRM 路徑規劃**的即時旅行景點推薦系統。

## 系統特點

### 🎯 核心功能

1. **即時沿途推薦**
   - 在旅行途中動態推薦沿途景點和餐廳
   - 基於用戶當前位置和目的地

2. **智能路徑規劃**
   - 使用 OSRM Server 進行精確路徑規劃
   - 計算繞道成本和時間影響

3. **個性化推薦**
   - 基於用戶歷史偏好
   - DLRM 深度學習模型融合多種特徵

4. **推薦理由生成**
   - 自動生成推薦理由
   - 包括評分、熱門度、繞道時間等

### 🏗️ 系統架構

```
travel_recommender/
├── dlrm_model.py              # DLRM 深度學習推薦模型
├── data_processor.py          # 數據預處理模組
├── route_aware_recommender.py # 路徑感知推薦引擎
├── train_model.py             # 模型訓練腳本
├── example_usage.py           # 使用範例
└── requirements.txt           # 依賴套件
```

## 快速開始

### 1. 安裝依賴

```bash
cd travel_recommender
pip install -r requirements.txt
```

### 2. 準備數據

確保數據文件位於正確位置：
```
datasets/
├── meta-other.json    # POI 元數據
└── review-other.json  # 用戶評論數據
```

### 3. 訓練模型

```bash
python train_model.py \
    --meta-path ../datasets/meta-other.json \
    --review-path ../datasets/review-other.json \
    --max-pois 10000 \
    --max-reviews 50000 \
    --epochs 20 \
    --batch-size 256
```

訓練參數：
- `--embedding-dim`: 嵌入維度 (default: 64)
- `--epochs`: 訓練輪數 (default: 20)
- `--batch-size`: 批次大小 (default: 256)
- `--learning-rate`: 學習率 (default: 0.001)

### 4. 使用推薦系統

```python
from route_aware_recommender import create_route_recommender

# 初始化推薦器
recommender = create_route_recommender(
    poi_data_path="datasets/meta-other.json",
    model_checkpoint="models/travel_dlrm.pth"
)

# 設定旅行路線 (金門大橋 → 迪士尼樂園)
start_location = (37.8199, -122.4783)  # 金門大橋 (舊金山)
end_location = (33.8121, -117.9190)    # 迪士尼樂園 (安那罕)

# 用戶歷史記錄
user_history = [
    {'category': 'cafe', 'rating': 5.0},
    {'category': 'museum', 'rating': 4.5}
]

# 獲取推薦
recommendations = recommender.recommend_on_route(
    user_id="user_001",
    user_history=user_history,
    start_location=start_location,
    end_location=end_location,
    top_k=5
)

# 查看推薦結果
for rec in recommendations:
    print(f"{rec['poi']['name']}: {rec['score']:.3f}")
    print(f"  額外時間: {rec['extra_time_minutes']:.0f} 分鐘")
    print(f"  理由: {rec['reasons']}")
```

### 5. 運行範例

```bash
python example_usage.py
```

這將運行多個範例：
- 即時路線推薦
- 地點搜索
- 用戶偏好分析
- 路徑規劃與繞道分析

## DLRM 模型架構

```
用戶特徵              POI特徵               路徑特徵
    ↓                    ↓                    ↓
[BottomMLP]        [BottomMLP]          [BottomMLP]
    ↓                    ↓                    ↓
[Embedding]        [Embedding]          [Embedding]
    ↓                    ↓                    ↓
    └──────────────┬─────────────┬──────────┘
                   ↓             ↓
          [Feature Interaction]  [路徑注意力]
                   ↓             ↓
                [TopMLP]
                   ↓
              [評分輸出]
```

### 特徵設計

**用戶特徵**：
- 連續特徵: 平均評分、評分標準差、活躍度
- 類別特徵: 年齡組、性別、旅行風格

**POI特徵**：
- 連續特徵: 評分、評論數、價格、地理位置
- 類別特徵: 類別、城市、價格等級

**路徑特徵**：
- 額外距離、額外時間、繞道比例

## API 服務 (規劃中)

### 啟動服務

```bash
cd api
make run
```

### API 端點

```
POST /api/v1/recommend
{
    "user_id": "user_001",
    "start": {"lat": 37.8199, "lon": -122.4783},
    "end": {"lat": 33.8121, "lon": -117.9190},
    "top_k": 5
}
```

## 系統配置

### OSRM Server

默認使用公共 OSRM 服務器：
```
http://router.project-osrm.org
```

如需使用私有服務器，請修改 `OSRMClient` 初始化：

```python
osrm = OSRMClient(server_url="http://your-osrm-server:5000")
```

### 模型參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `embedding_dim` | 64 | 嵌入維度 |
| `bottom_mlp_dims` | [256, 128] | 底層 MLP 維度 |
| `top_mlp_dims` | [512, 256, 128] | 頂層 MLP 維度 |
| `dropout` | 0.2 | Dropout 率 |

## 評估指標

- **AUC**: Area Under ROC Curve
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: 前 K 個推薦的精確率
- **Recall@K**: 前 K 個推薦的召回率
- **Hit Rate@K**: 命中率

## 性能優化

### 快取策略

1. **路徑查詢快取**
   - 使用 `lru_cache` 快取 OSRM 查詢結果
   - 減少重複的 API 請求

2. **POI 編碼快取**
   - 預先編碼所有 POI
   - 避免重複的特徵提取

### 批次處理

```python
# 批次評分多個 POI
scores = recommender._score_pois(
    user_profile, 
    candidate_pois,
    start_location,
    end_location
)
```

## 故障排除

### 問題 1: OSRM 連接失敗

**解決方案**：
- 檢查網絡連接
- 使用本地 OSRM 服務器
- 增加請求超時時間

### 問題 2: 記憶體不足

**解決方案**：
- 減少 `max_pois` 和 `max_reviews`
- 降低批次大小 `batch_size`
- 使用 CPU 而非 GPU

### 問題 3: 推薦結果為空

**解決方案**：
- 增加 `max_detour_ratio`
- 延長 `max_extra_duration`
- 擴大候選 POI 搜索範圍

## 開發路線圖

- [x] DLRM 模型實現
- [x] 路徑感知推薦引擎
- [x] 數據處理模組
- [x] 訓練腳本
- [ ] Go API 服務整合
- [ ] 實時位置追蹤
- [ ] 多人協同旅行推薦
- [ ] 推薦結果可視化
- [ ] 移動端 APP

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 授權

MIT License

## 聯繫方式

如有問題，請提交 Issue 或聯繫開發團隊。
