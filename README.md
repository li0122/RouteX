# RouteX - 智能旅遊推薦系統

> 基於深度學習的路徑感知個性化旅遊推薦系統  
> **DLRM + OSRM + LLM** 三位一體，提供智能行程規劃與景點推薦

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 目錄

- [系統概述](#系統概述)
- [核心功能](#核心功能)
- [系統架構](#系統架構)
- [快速開始](#快速開始)
- [詳細使用指南](#詳細使用指南)
- [API 文檔](#api-文檔)
- [模型訓練](#模型訓練)
- [評估與測試](#評估與測試)
- [部署指南](#部署指南)
- [技術細節](#技術細節)
- [常見問題](#常見問題)

---

## 🎯 系統概述

**RouteX** 是一個基於深度學習的智能旅遊推薦系統，結合了：
- 🧠 **DLRM (Deep Learning Recommendation Model)** - 個性化推薦引擎
- 🗺️ **OSRM (Open Source Routing Machine)** - 精確路徑規劃
- 🤖 **LLM (Large Language Model)** - 智能意圖理解與過濾

### 特色亮點

✨ **路徑感知推薦** - 不僅推薦好的景點，更推薦「順路」的景點  
✨ **個性化畫像** - 基於用戶偏好、評分習慣、預算等多維度特徵  
✨ **智能行程規劃** - 自動優化景點訪問順序，最小化繞道成本  
✨ **LLM 語意過濾** - 理解用戶活動意圖，精準過濾相關景點  
✨ **Web 可視化介面** - 互動式地圖展示推薦路線與景點

---

## 🚀 核心功能

### 1️⃣ 路徑感知推薦 (Route-Aware Recommendation)

在用戶指定的起點和終點之間，智能推薦沿途景點：

```python
recommendations = recommender.recommend_on_route(
    user_id='user_001',
    user_history=[
        {'category': 'Restaurant', 'rating': 5.0},
        {'category': 'Museum', 'rating': 4.5}
    ],
    start_location=(37.7749, -122.4194),  # 舊金山
    end_location=(34.0522, -118.2437),    # 洛杉磯
    activityIntent="美食探索",             # 活動意圖
    top_k=10,                              # 推薦數量
    max_detour_ratio=1.3,                 # 最大繞道比例
    max_extra_duration=900                # 最大額外時間(秒)
)
```

**特點**：
- ✅ 結合 DLRM 評分與路徑代價
- ✅ 動態計算繞道時間和距離
- ✅ LLM 語意過濾確保推薦相關性

### 2️⃣ 完整行程規劃 (Itinerary Planning)

生成優化後的完整旅遊行程：

```python
itinerary = recommender.recommend_itinerary(
    user_id='user_001',
    user_history=user_history,
    start_location=(37.7749, -122.4194),
    end_location=(34.0522, -118.2437),
    activityIntent="文化之旅",
    time_budget=240,  # 總時間預算(分鐘)
    top_k=20          # 候選景點數
)
```

**輸出**：
- 📍 優化後的景點訪問順序
- ⏱️ 每個景點的預估停留時間
- 🛣️ 完整的導航路線
- 💡 個性化推薦理由

### 3️⃣ 用戶畫像推薦 (Profile-Based Recommendation)

基於用戶畫像生成個性化推薦：

```python
# Web API 調用
POST /api/recommend_by_profile
{
  "user_profile": {
    "avg_rating": 4.2,      // 平均評分標準
    "rating_std": 0.6,      // 評分變異度
    "num_reviews": 50,      // 評論數量
    "budget": 3             // 預算等級 (1-5)
  },
  "filters": {
    "categories": ["Restaurant", "Museum"],
    "state": "California",
    "price_range": [2, 4]   // 價格範圍
  },
  "top_k": 20
}
```

### 4️⃣ 智能地圖介面 (Interactive Web UI)

- 🗺️ **Leaflet 地圖視覺化** - 拖放設定起終點
- 📍 **即時路線預覽** - 動態顯示推薦景點
- 🎨 **景點詳情卡片** - 評分、類別、價格、推薦理由
- 🔧 **參數調整面板** - 繞道比例、活動意圖、類別篩選

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                        Web 層 (Flask)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  地圖介面    │  │  用戶畫像    │  │  行程規劃    │     │
│  │  (Leaflet)   │  │  設定頁面    │  │  視覺化      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓ REST API
┌─────────────────────────────────────────────────────────────┐
│                   推薦引擎層 (Python)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RouteAwareRecommender (核心推薦器)                  │  │
│  │  ├─ DLRM 模型 (個性化評分)                          │  │
│  │  ├─ 空間索引 (KD-Tree 快速檢索)                     │  │
│  │  ├─ 路徑計算 (OSRM Client)                          │  │
│  │  └─ LLM 過濾器 (語意相關性審核)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   資料處理層 (Data Pipeline)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ POI 處理器   │  │ 評論處理器   │  │ 特徵編碼器   │     │
│  │ (515K POIs)  │  │ (50K Reviews)│  │ (Category)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   外部服務層 (APIs)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ OSRM Server  │  │ OpenAI API   │  │ Dataset      │     │
│  │ (路徑規劃)  │  │ (LLM 服務)   │  │ (California) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 核心模組說明

| 模組 | 檔案 | 功能 |
|------|------|------|
| **DLRM 模型** | `dlrm_model.py` | 深度學習推薦模型，特徵交互與評分 |
| **路徑感知推薦器** | `route_aware_recommender.py` | 結合 DLRM 和 OSRM 的推薦引擎 |
| **LLM 過濾器** | `simple_llm_filter.py` | 基於 OpenAI API 的語意過濾 |
| **資料處理器** | `data_processor.py` | POI 和評論資料的載入與預處理 |
| **Web 服務** | `web_app.py` | Flask API 與前端介面 |
| **模型訓練** | `train_model.py` | DLRM 模型訓練腳本 |
| **評估工具** | `evaluate_metrics.py` | 推薦品質評估 (Precision, NDCG, AUC) |

---

## ⚡ 快速開始

### 系統需求

- **Python**: 3.8+
- **GPU**: CUDA 11.8+ (可選，用於訓練加速)
- **記憶體**: 8GB+ RAM
- **儲存**: 10GB+ 可用空間

### 1. 克隆專案

```bash
git clone https://github.com/your-repo/RouteX.git
cd RouteX
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

**核心依賴**：
```
torch>=2.0.0
flask>=2.0.0
flask-cors>=4.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
requests>=2.26.0
tqdm>=4.62.0
openai>=1.0.0
```

### 3. 準備資料集

確保資料集位於正確位置：

```
RouteX/
└── datasets/
    ├── meta-California.json.gz      # POI 元資料 (515,961 個)
    └── review-California.json.gz    # 用戶評論 (50,000 條)
```

**資料格式**：
- **POI**: `gmap_id`, `name`, `latitude`, `longitude`, `category`, `avg_rating`, `num_of_reviews`, `price`, `state`, `address`
- **Review**: `user_id`, `gmap_id`, `rating`, `text`, `time`

### 4. 訓練模型（可選）

如果已有預訓練模型 `models/travel_dlrm.pth`，可跳過此步驟。

```bash
python train_model.py \
    --meta-path datasets/meta-California.json.gz \
    --review-path datasets/review-California.json.gz \
    --max-pois 10000 \
    --max-reviews 50000 \
    --epochs 63 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --checkpoint-path models/travel_dlrm.pth
```

**訓練統計**：
- 訓練時間：約 6-8 小時 (63 epochs, GPU)
- 模型大小：約 200MB
- 最佳 AUC：0.86+

### 5. 啟動 Web 服務

```bash
python web_app.py
```

服務將在 `http://localhost:5000` 啟動。

**可用頁面**：
- 🏠 首頁：`http://localhost:5000/`
- 🗺️ 地圖測試：`http://localhost:5000/test_leaflet`
- 👤 用戶畫像：`http://localhost:5000/profile`
- 📊 系統狀態：`http://localhost:5000/api/status`

---

## 📖 詳細使用指南

### Python API 使用

#### 基礎推薦

```python
from route_aware_recommender import create_route_recommender

# 1. 初始化推薦器
recommender = create_route_recommender(
    poi_data_path='datasets/meta-California.json.gz',
    model_checkpoint='models/travel_dlrm.pth',
    device='cuda',                # 使用 GPU
    enable_spatial_index=True,    # 啟用空間索引加速
    enable_async=False            # Web 環境使用同步模式
)

# 2. 設定 OSRM 客戶端
from route_aware_recommender import OSRMClient
recommender.osrm_client = OSRMClient(
    server_url="http://140.125.32.60:5000"
)

# 3. 準備用戶歷史
user_history = [
    {'category': 'Restaurant', 'rating': 5.0},
    {'category': 'Cafe', 'rating': 4.5},
    {'category': 'Museum', 'rating': 4.0}
]

# 4. 生成推薦
recommendations = recommender.recommend_on_route(
    user_id='user_001',
    user_history=user_history,
    start_location=(37.7749, -122.4194),  # 舊金山
    end_location=(34.0522, -118.2437),    # 洛杉磯
    activityIntent="美食之旅",
    top_k=10
)

# 5. 查看結果
for i, rec in enumerate(recommendations):
    poi = rec['poi']
    print(f"{i+1}. {poi['name']}")
    print(f"   類別: {poi['primary_category']}")
    print(f"   評分: {poi['avg_rating']:.1f} ⭐ ({poi['num_reviews']} 評論)")
    print(f"   額外時間: {rec['extra_time_minutes']:.0f} 分鐘")
    print(f"   LLM 審核: {'✓' if rec['llm_approved'] else '✗'}")
    print(f"   推薦理由: {', '.join(rec['reasons'])}")
    print()
```

#### 完整行程規劃

```python
itinerary = recommender.recommend_itinerary(
    user_id='user_001',
    user_history=user_history,
    start_location=(37.7749, -122.4194),
    end_location=(34.0522, -118.2437),
    activityIntent="文化探索",
    time_budget=300,  # 5 小時
    top_k=20
)

print(f"行程總覽:")
print(f"  景點數量: {itinerary['total_stops']}")
print(f"  總時長: {itinerary['total_duration']} 分鐘")
print(f"  總距離: {itinerary['total_distance']:.1f} 公里")
print(f"\n景點列表:")
for stop in itinerary['itinerary']:
    print(f"  {stop['order']}. {stop['poi']['name']}")
    print(f"     停留時間: {stop['estimated_duration']} 分鐘")
```

### Web API 使用

#### 1. 路徑推薦 API

**端點**: `POST /api/recommend`

**請求**:
```json
{
  "start_location": [37.7749, -122.4194],
  "end_location": [34.0522, -118.2437],
  "activity_intent": "美食探索",
  "categories": ["Restaurant", "Cafe"],
  "top_k": 10
}
```

**響應**:
```json
{
  "success": true,
  "count": 10,
  "recommendations": [
    {
      "poi": {
        "name": "French Laundry",
        "primary_category": "Restaurant",
        "avg_rating": 4.8,
        "num_reviews": 2500,
        "latitude": 38.4024,
        "longitude": -122.3635
      },
      "score": 0.92,
      "extra_time_minutes": 15,
      "llm_approved": true,
      "reasons": ["高評分餐廳", "符合活動意圖", "繞道時間短"]
    }
  ],
  "processing_time": 2.5
}
```

#### 2. 用戶畫像推薦 API

**端點**: `POST /api/recommend_by_profile`

**請求**:
```json
{
  "user_profile": {
    "avg_rating": 4.2,
    "rating_std": 0.5,
    "num_reviews": 30,
    "budget": 3
  },
  "filters": {
    "categories": ["Museum", "Park"],
    "state": "California",
    "price_range": [1, 3]
  },
  "top_k": 20
}
```

#### 3. 完整行程 API

**端點**: `POST /api/itinerary`

**請求**:
```json
{
  "start": [37.7749, -122.4194],
  "end": [34.0522, -118.2437],
  "activity_intent": "親子遊",
  "time_budget": 240,
  "top_k": 15
}
```

---

## 🔧 模型訓練

### 訓練資料準備

```python
from data_processor import POIDataProcessor, ReviewDataProcessor

# 載入 POI 資料
poi_processor = POIDataProcessor('datasets/meta-California.json.gz')
poi_processor.load_data(max_records=10000)
poi_processor.preprocess()

# 載入評論資料
review_processor = ReviewDataProcessor('datasets/review-California.json.gz')
review_processor.load_data(max_records=50000)
review_processor.preprocess()
```

### 模型配置

**DLRM 架構參數**：

```python
model = create_travel_dlrm(
    user_continuous_dim=10,        # 用戶連續特徵維度
    poi_continuous_dim=8,          # POI 連續特徵維度
    path_continuous_dim=4,         # 路徑連續特徵維度
    user_vocab_sizes={},           # 用戶類別特徵詞彙表
    poi_vocab_sizes={              # POI 類別特徵詞彙表
        'category': 101,           #   - 類別數量
        'state': 3097,             #   - 州/地區數量
        'price_level': 5           #   - 價格等級數量
    },
    embedding_dim=64,              # 嵌入向量維度
    bottom_mlp_dims=[256, 128],    # Bottom MLP 隱藏層
    top_mlp_dims=[512, 256, 128],  # Top MLP 隱藏層
    dropout=0.2                    # Dropout 比例
)
```

**訓練超參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `epochs` | 63 | 訓練輪數 |
| `batch_size` | 256 | 批次大小 |
| `learning_rate` | 0.001 | 初始學習率 |
| `weight_decay` | 1e-5 | L2 正則化 |
| `negative_ratio` | 4 | 負樣本比例 (1:4) |
| `optimizer` | Adam | 優化器 |
| `scheduler` | ReduceLROnPlateau | 學習率調度器 |

### 完整訓練流程

```bash
# 使用 GPU 訓練
python train_model.py \
    --meta-path datasets/meta-California.json.gz \
    --review-path datasets/review-California.json.gz \
    --max-pois 10000 \
    --max-reviews 50000 \
    --negative-ratio 4 \
    --embedding-dim 64 \
    --epochs 63 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --weight-decay 1e-5 \
    --checkpoint-path models/travel_dlrm.pth \
    --processor-path models/poi_processor.pkl
```

**訓練監控**：
```
Epoch 1/63: Loss=0.6542, Val AUC=0.6234
Epoch 10/63: Loss=0.4521, Val AUC=0.7456
Epoch 30/63: Loss=0.3125, Val AUC=0.8123
Epoch 63/63: Loss=0.2834, Val AUC=0.8634
✓ 最佳模型已保存
```

---

## 📊 評估與測試

### 評估指標

使用 `evaluate_metrics.py` 計算推薦品質指標：

```bash
python evaluate_metrics.py \
    --model models/travel_dlrm.pth \
    --poi-data datasets/meta-California.json.gz \
    --review-data datasets/review-California.json.gz \
    --k-values 1 3 10 \
    --max-users 500
```

**評估指標**：

| 指標 | K值 | 說明 |
|------|-----|------|
| **Precision@K** | 1, 3, 10 | 推薦的前 K 個中有多少是用戶喜歡的 |
| **Recall@K** | 1, 3, 10 | 用戶喜歡的景點中有多少被推薦到前 K |
| **F1-Score@K** | 1, 3, 10 | Precision 和 Recall 的調和平均 |
| **NDCG@K** | 1, 3, 10 | 考慮排序位置的推薦品質 |
| **AUC** | - | ROC 曲線下面積，整體分類能力 |

**典型結果**（63 epochs 訓練）：

```
評估結果
==========================================================
K = 10:
  Precision@10: 0.XXXX
  Recall@10:    0.XXXX
  F1-Score@10:  0.XXXX
  NDCG@10:      0.XXXX

整體指標:
  AUC: 0.86XX
==========================================================
```

### 基準模型對比

使用 `evaluate_baselines.py` 評估基準模型：

```bash
python evaluate_baselines.py \
    --poi-data datasets/meta-California.json.gz \
    --review-data datasets/review-California.json.gz \
    --max-users 500 \
    --models random popularity cf mf
```

**對比結果**：

| 模型 | Precision@10 | Recall@10 | NDCG@10 | AUC |
|------|--------------|-----------|---------|-----|
| Random | 0.01XX | 0.0XXX | 0.1XXX | 0.500 |
| Popularity | 0.0XXX | 0.0XXX | 0.2XXX | 0.6XX |
| Collaborative Filtering | 0.0XXX | 0.0XXX | 0.3XXX | 0.7XX |
| Matrix Factorization | 0.0XXX | 0.0XXX | 0.3XXX | 0.7XX |
| **RouteX (DLRM)** | **0.XXXX** | **0.XXXX** | **0.XXXX** | **0.86XX** |

---

## 🚢 部署指南

### Docker 部署（推薦）

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "web_app.py"]
```

```bash
# 建立映像
docker build -t routex:latest .

# 執行容器
docker run -d \
  --name routex \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/datasets:/app/datasets \
  routex:latest
```

### 生產環境配置

使用 `gunicorn` 部署：

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 \
  --timeout 120 \
  --worker-class sync \
  web_app:app
```

### OSRM Server 部署

```bash
# 下載 OSM 資料
wget http://download.geofabrik.de/north-america/us/california-latest.osm.pbf

# 預處理資料
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract \
  -p /opt/car.lua /data/california-latest.osm.pbf

docker run -t -v $(pwd):/data osrm/osrm-backend osrm-contract \
  /data/california-latest.osrm

# 啟動 OSRM 服務
docker run -t -p 5000:5000 -v $(pwd):/data \
  osrm/osrm-backend osrm-routed \
  --algorithm mld /data/california-latest.osrm
```

---

## 🔬 技術細節

### DLRM 模型架構

```
輸入層:
├─ 用戶連續特徵 (10維): [avg_rating, rating_std, num_reviews, ...]
├─ 用戶類別特徵: {} (當前模型未使用)
├─ POI 連續特徵 (8維): [avg_rating, num_reviews, price, lat, lng, ...]
├─ POI 類別特徵: {category, state, price_level}
└─ 路徑連續特徵 (4維): [extra_distance, extra_time, detour_ratio, ...]

Bottom MLP:
├─ 用戶 MLP: [10] → [256] → [128]
├─ POI MLP:  [8]  → [256] → [128]
└─ 路徑 MLP: [4]  → [128] → [64]

Embedding 層:
├─ Category Embedding: [101] → [64]
├─ State Embedding:    [3097] → [64]
└─ Price Embedding:    [5] → [64]

Feature Interaction:
├─ 點積交互 (Dot Product)
├─ 注意力機制 (Attention)
└─ 特徵組合 (Concatenation)

Top MLP:
[Interaction Features] → [512] → [256] → [128] → [1]

輸出層:
└─ Sigmoid(logits) → Click Probability [0, 1]
```

### 路徑計算演算法

**繞道成本計算**：

```python
# 1. 原始路線距離
d_direct = OSRM.route(start, end).distance

# 2. 繞道路線距離
d_detour = OSRM.route(start, poi).distance + \
           OSRM.route(poi, end).distance

# 3. 額外距離
extra_distance = d_detour - d_direct

# 4. 繞道比例
detour_ratio = d_detour / d_direct

# 5. 額外時間
extra_time = (d_detour - d_direct) / avg_speed
```

**過濾條件**：
- `detour_ratio < max_detour_ratio` (預設 1.3)
- `extra_time < max_extra_duration` (預設 900秒)

### 空間索引優化

使用 **KD-Tree** 加速空間檢索：

```python
from scipy.spatial import cKDTree

# 建立 KD-Tree
coords = np.array([(poi['latitude'], poi['longitude']) 
                   for poi in all_pois])
kd_tree = cKDTree(coords)

# 範圍查詢 (半徑 50km)
poi_indices = kd_tree.query_ball_point(
    (current_lat, current_lng),
    r=50/111  # 緯度每度約 111km
)

# 複雜度: O(log N) vs O(N)
```

### LLM 語意過濾

使用 **OpenAI GPT-4** 進行語意相關性判斷：

```python
prompt = f"""
用戶活動意圖: {activity_intent}
景點名稱: {poi_name}
景點類別: {poi_category}
景點描述: {poi_description}

問題: 這個景點是否適合用戶的活動意圖？
要求: 只回答 'YES' 或 'NO'，並簡短說明理由(不超過20字)。
"""

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    max_tokens=50
)
```

---

## ❓ 常見問題

### Q1: 為什麼推薦結果為空？

**可能原因**：
1. ✅ `max_detour_ratio` 太小 → 增加至 1.5
2. ✅ `max_extra_duration` 太短 → 增加至 1800秒
3. ✅ 類別過濾太嚴格 → 放寬 `categories`
4. ✅ 起終點距離太遠 → 選擇較近的目的地

### Q2: OSRM 連接失敗怎麼辦？

**解決方案**：
```python
# 1. 檢查 OSRM 服務狀態
curl http://140.125.32.60:5000/

# 2. 使用本地 OSRM 服務
osrm_client = OSRMClient(server_url="http://localhost:5000")

# 3. 增加超時時間
response = requests.get(url, timeout=30)
```

### Q3: 模型預測分數都是負數？

**原因**: 模型輸出的是 **logits（未經 sigmoid 的原始分數）**

**解決**: 排序時直接使用 logits，不要 sigmoid：
```python
# ✅ 正確：用 logits 排序
scores = model(features)  # [-644, -29482, -30674]
sorted_pois = sorted(zip(pois, scores), key=lambda x: x[1], reverse=True)

# ❌ 錯誤：sigmoid 後失去區分度
scores = torch.sigmoid(model(features))  # [0.000, 0.000, 0.000]
```

### Q4: GPU 記憶體不足？

**解決方案**：
1. 減少批次大小：`--batch-size 128`
2. 降低嵌入維度：`--embedding-dim 32`
3. 使用 CPU 訓練：`--device cpu`
4. 啟用混合精度：`torch.cuda.amp.autocast()`

### Q5: 如何自訂活動意圖？

```python
# 預定義意圖
活動意圖範例 = [
    "美食探索",
    "文化之旅",
    "親子遊",
    "戶外冒險",
    "購物娛樂",
    "放鬆度假"
]

# 自訂意圖
recommendations = recommender.recommend_on_route(
    activityIntent="尋找小眾咖啡館與獨立書店",
    ...
)
```

---

## 📚 相關文件

- 📄 [OVERVIEW.md](OVERVIEW.md) - 系統概述
- 📄 [DELIVERY.md](DELIVERY.md) - 交付說明
- 📄 [GPU_ACCELERATION.md](GPU_ACCELERATION.md) - GPU 加速指南
- 📄 [MEMORY_EFFICIENT_TRAINING.md](MEMORY_EFFICIENT_TRAINING.md) - 記憶體優化
- 📄 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - 評估指南

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

**開發規範**：
1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

---

## 📝 授權

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

## 📧 聯絡方式

- **專案維護者**: RouteX Team
- **Email**: your-email@example.com
- **GitHub Issues**: [提交問題](https://github.com/your-repo/RouteX/issues)

---

## 🎓 論文引用

如果本專案對您的研究有幫助，請引用：

```bibtex
@misc{routex2025,
  title={RouteX: A Route-Aware Deep Learning Recommendation System for Travel Planning},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-repo/RouteX}
}
```

---

## 🌟 致謝

- **PyTorch** - 深度學習框架
- **OSRM** - 開源路徑規劃引擎
- **OpenAI** - GPT-4 API 支持
- **Google Maps** - POI 資料來源
- **Flask** - Web 框架

---

<div align="center">

**⭐ 如果這個專案對你有幫助，請給個 Star！⭐**

Made with ❤️ by RouteX Team

</div>
