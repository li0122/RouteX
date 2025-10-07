# 旅行推薦系統架構說明 (model_structure)

本文件彙整 `travel_recommender/rec_sys` 專案的整體運作模式，涵蓋資料流、訓練流程、模型架構與推論管線，便於快速理解與後續維護。

---

## 1. 系統總覽

```
┌─────────────────────────────────────────────────────────────┐
│                         用戶 / API                          │
│                 (CLI、Go / FastAPI 服務規劃中)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RouteAwareRecommender                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  用戶偏好建模  │  OSRM 路徑評估  │  TravelDLRM 模型 │ │
│  │  (history→profile)│  (detour計算) │  (深度推薦)      │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Processors 層                      │
│  POIDataProcessor  │  ReviewDataProcessor  │  Cache/Encode │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      datasets/ (JSON, JSON.GZ)              │
│  meta-*.json(.gz)  │  review-*.json(.gz)  │  其他資源       │
└─────────────────────────────────────────────────────────────┘
```

核心模組對應檔案：

- `route_aware_recommender.py`: 路徑感知推薦引擎與 OSRM 客戶端。
- `dlrm_model.py`: 旅行版 DLRM 模型與損失函式。
- `data_processor.py`: POI / 評論預處理、特徵編碼、地理搜尋。
- `train_model.py`: 訓練腳本、資料集封裝、評估邏輯。
- `example_usage.py`: 使用場景示範 (推薦、地點搜尋、路徑分析)。
- `quick_test.py`, `verify_datasets.py`, `quickstart.sh`: 運維與驗證工具。

---

## 2. 資料與預處理

### 2.1 資料格式

- `datasets/meta-*.json(.gz)`: 每行一筆 POI 元資料 (經緯度、評分、類別、營業資訊)。
- `datasets/review-*.json(.gz)`: 用戶評論紀錄 (user_id、rating、text、gmap_id)。
- `SUMMARY.txt`, `DATASETS_READY.md`: 整備紀錄與使用指南。

### 2.2 POIDataProcessor (`data_processor.py`)

主要職責：

1. **載入多格式**：自動辨識 `.json` 及 `.json.gz`，支援分批讀取提示。
2. **欄位萃取**：標準化座標、評分、價格、營業時間等欄位。
3. **特徵編碼**：
   - 連續特徵：評分、評論數、價格、地理位置、是否 24h 等。
   - 類別特徵：`primary_category`、`state`、`price_level` 轉換為索引。
4. **地理搜尋**：`get_pois_by_location` 透過 Haversine 公式提供半徑搜尋。
5. **統計資訊**：彙整類別分佈、平均評分等供分析使用。

輸出格式範例：

```
{
  'continuous': np.ndarray(shape=(8,)),
  'categorical': {'category': idx, 'state': idx, 'price_level': level},
  'raw': 原始 POI 字典
}
```

### 2.3 ReviewDataProcessor (`data_processor.py`)

- 按 `user_id`、`gmap_id` 建立倒排索引。
- 供訓練資料生成器產生正負樣本，並提供 `get_user_profile` 建立基本畫像。
- 預留 `create_user_poi_pairs` 擴充負樣本策略。

---

## 3. 模型層 (`dlrm_model.py`)

TravelDLRM 由 DLRM 概念延伸，整合用戶、POI、路徑三類特徵：

```
連續特徵 ─▶ BottomMLP (BatchNorm + ReLU + Dropout)
類別特徵 ─▶ EmbeddingLayer (nn.Embedding)
                    │
                    ▼
            FeatureInteraction (雙兩兩點積)
                    │
                    ▼
              TopMLP (多層全連接)
                    │
                    ▼
               Sigmoid 評分輸出
```

關鍵元件：

- `EmbeddingLayer`: 為每個類別特徵建立獨立嵌入向量。
- `BottomMLP`: 連續特徵映射至嵌入維度 (預設 64)。
- `FeatureInteraction`: 將所有嵌入向量進行 pairwise dot-product 交互。
- `TopMLP`: 將交互結果與原始嵌入拼接後輸出預測分數。
- `TravelDLRM`: 提供 `predict` (eval) 與帶注意力權重的 `forward` 輸出。
- `DLRMLoss`: 支援 BCE / BPR / Margin Loss (訓練腳本目前使用 BCE with logits)。

### 預設維度設定

| 區塊            | 預設維度                         |
|-----------------|----------------------------------|
| 嵌入維度        | `embedding_dim=64`              |
| Bottom MLP      | `[256, 128]`                    |
| Top MLP         | `[512, 256, 128]`               |
| Dropout         | `0.2~0.3`                       |
| 用戶連續特徵    | 10 維 (均值評分、活躍度等)        |
| POI 連續特徵    | 8 維 (評分、評論數、價格等)       |
| 路徑連續特徵    | 4 維 (額外距離、時間、比率等)     |

---

## 4. 訓練流程 (`train_model.py`)

### 4.1 數據載入

- `load_and_process_data`: 一次性讀取 (適合中型資料)。
- `load_data_in_shards`: 分片載入評論 (`shard_size` 預設 100000)，降低記憶體壓力。

### 4.2 資料集與 DataLoader

- `TravelRecommendDataset`：
  - 記憶體高效模式下僅保留索引。
  - 正樣本：`rating >= 4` 的用戶-POI 互動。
  - 負樣本：隨機抽取未互動 POI (比例 `negative_ratio`，預設 4)。
  - 返回批次字典包含連續與類別特徵張量。
- `collate_fn`: 將 numpy 轉換為張量並整理批次格式。
- DataLoader 分割為 80/20 訓練與驗證。

### 4.3 訓練設定

- 優化器：Adam (`lr=1e-3`, `weight_decay=1e-5`)。
- Scheduler：`ReduceLROnPlateau` 監控 AUC。
- 指標：AUC、Accuracy、Precision、Recall (使用 scikit-learn)。
- Gradient clipping：`max_norm=1.0`。

### 4.4 輸出

- 最佳模型權重：`models/travel_dlrm.pth` (或自訂 `--checkpoint-path`)。
- 編碼器狀態：`poi_processor.save()` 輸出 `models/poi_processor.pkl`。
- 訓練日誌印出 epoch 耗時與指標。

### 4.5 記憶體化工具

- `MEMORY_EFFICIENT_TRAINING.md`, `train_with_progress.py`, `train_full_california.sh` 提供進階訓練建議。

---

## 5. 推論與推薦流程 (`route_aware_recommender.py`)

### 5.1 OSRM 路徑客戶端

- `OSRMClient` 透過 REST API 查詢路線，預設公共伺服器 `http://router.project-osrm.org`。
- `lru_cache` 快取查詢結果 (最大 1000 筆)。
- `calculate_detour` 計算直達與繞道距離/時間，輸出 detour ratio。
- `is_poi_on_route` 依 `max_detour_ratio`、`max_extra_duration` 過濾候選點。

### 5.2 用戶偏好建模

- `UserPreferenceModel`：
  - 從歷史紀錄統計平均評分、偏好類別、活躍度。
  - `get_user_features` 產生 10 維連續特徵 (標準化後供模型輸入)。

### 5.3 RouteAwareRecommender

流程：

1. 建立用戶畫像 (`build_user_profile`)。
2. 取得候選 POI：
   - 若未指定，自動以起終點中點搜尋 `radius_km` (預設 50km)。
3. 透過 OSRM 過濾合理繞道的 POI。
4. 編碼 POI / 路徑特徵並批次送入 `TravelDLRM.predict`。
5. 依分數排序，回傳 top-k 結果。
6. ` _generate_recommendation_reasons` 產生最多三個可解釋理由 (高評分、熱門、符合偏好、繞道時間等)。

輸出結果格式：

```
{
  'poi': POI 原始資料,
  'score': 浮點評分,
  'detour_info': {...},
  'reasons': ["⭐ 高評分...", ...],
  'extra_time_minutes': 浮點分鐘數
}
```

---

## 6. 支援腳本與工具

| 檔案                     | 功能摘要 |
|--------------------------|----------|
| `example_usage.py`       | 提供五個範例 (即時推薦、地點搜尋、偏好分析、路徑分析、批量推薦)。 |
| `quick_test.py`          | 環境與資料檢查，快速驗證核心模組可用性。 |
| `verify_datasets.py`     | 自動檢查 datasets 目錄，提供使用建議。 |
| `quickstart.sh` / `start.sh` | 互動式 CLI，整合安裝、訓練、測試選項。 |
| `train_with_progress.py` | 包裝訓練腳本並顯示整體進度。 |
| `train_full_california.sh` | 為完整 California 資料集預設訓練流程。 |
| `MEMORY_EFFICIENT_TRAINING.md` | 大型資料訓練注意事項。 |

---

## 7. 執行指令速查

### 7.1 環境與驗證

```bash
pip install -r requirements.txt
python quick_test.py
python verify_datasets.py
```

### 7.2 訓練示例

```bash
# 小型測試
python train_model.py \
  --meta-path datasets/meta-other.json \
  --review-path datasets/review-other.json \
  --max-pois 1000 --max-reviews 5000 \
  --epochs 5 --batch-size 128

# 完整訓練
python train_model.py \
  --meta-path datasets/meta-California.json.gz \
  --review-path datasets/review-California.json.gz \
  --max-pois 50000 --max-reviews 500000 \
  --epochs 20 --batch-size 64 \
  --memory-efficient --use-sharding --shard-size 100000
```

### 7.3 推薦使用

```python
from route_aware_recommender import create_route_recommender

recommender = create_route_recommender(
    poi_data_path="datasets/meta-other.json",
    model_checkpoint="models/travel_dlrm.pth"
)

recs = recommender.recommend_on_route(
  user_id="user_001",
  user_history=[{'category': 'cafe', 'rating': 5.0}],
  start_location=(37.8199, -122.4783),  # 金門大橋
  end_location=(33.8121, -117.9190),    # 迪士尼樂園
  top_k=5
)
```

---

## 8. 擴充與注意事項

- **OSRM 服務**：
  - 預設公共伺服器可能受限；可改 `OSRMClient(server_url=...)` 改用自建服務。
  - 建議增加重試或超時處理以提升穩定性。
- **特徵擴充**：
  - 可以在 `POIDataProcessor.encode_poi` 或 `UserPreferenceModel.get_user_features` 增加額外特徵，但需同步更新 `create_travel_dlrm` 的維度設定。
- **負樣本策略**：
  - 目前採隨機抽樣，可延伸為熱門度或地理距離感知的負樣本。
- **部署整合**：
  - 可將 `RouteAwareRecommender` 包裝為 FastAPI / Go 服務；README 中保留 API 計畫章節。
- **模型更新**：
  - 訓練完成後需同步保存 `poi_processor.pkl` 供推論時反序列化特徵編碼器。

---

## 9. 相關文檔索引

- `README.md`: 快速開始與操作指南。
- `OVERVIEW.md`: 架構圖與流程詳解 (與本文件內容互補)。
- `MEMORY_EFFICIENT_TRAINING.md`: 大型資料訓練策略。
- `DATASETS_READY.md`, `DATASET_MIGRATION.md`: 資料整備歷史。
- `DELIVERY.md`: 交付總結與後續規劃。

---

## 10. 後續建議

1. 依 `train_full_california.sh` 或 `quickstart.sh` 執行完整訓練並驗證評估指標。
2. 將 `RouteAwareRecommender` 封裝為 API 服務，結合行動端或行程規劃工具。
3. 針對推薦理由與 UI 呈現設計可視化，提升可解釋度。
4. 建立定期批次訓練流程 (資料更新 → 模型再訓練 → 評估 → 佈署)。

---

> 更新日期：2025-10-07  