#!/bin/bash

# 旅行推薦系統 - 快速開始腳本
# 資料集整理完成後的使用指南

set -e

echo "======================================================================"
echo "🚀 旅行推薦系統 - 快速開始"
echo "======================================================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 步驟 1: 驗證資料集
echo -e "${BLUE}步驟 1/4: 驗證資料集${NC}"
echo "----------------------------------------------------------------------"
python verify_datasets.py
echo ""

# 詢問使用者選擇
echo -e "${YELLOW}請選擇您要執行的操作：${NC}"
echo "  1) 快速測試（使用小型 other 資料集，約 5-10 分鐘）"
echo "  2) 完整訓練（使用 California 資料集，可能需要數小時）"
echo "  3) 僅驗證資料載入（不訓練模型）"
echo "  4) 查看資料集統計資訊"
echo "  5) 退出"
echo ""
read -p "請輸入選項 (1-5): " choice

case $choice in
  1)
    echo ""
    echo -e "${GREEN}=== 開始快速測試 ===${NC}"
    echo ""
    
    # 檢查是否已安裝依賴
    echo -e "${BLUE}步驟 2/4: 檢查依賴套件${NC}"
    echo "----------------------------------------------------------------------"
    python -c "import torch" 2>/dev/null || {
      echo -e "${YELLOW}未安裝 PyTorch，正在安裝依賴套件...${NC}"
      pip install -r requirements.txt
    }
    echo -e "${GREEN}✓ 依賴套件已就緒${NC}"
    echo ""
    
    # 測試資料載入
    echo -e "${BLUE}步驟 3/4: 測試資料載入${NC}"
    echo "----------------------------------------------------------------------"
    python -c "
from data_processor import POIDataProcessor
print('測試載入 POI 資料...')
p = POIDataProcessor('datasets/meta-other.json')
pois = p.load_data(max_records=100)
print(f'✓ 成功載入 {len(pois)} 個 POI')
"
    echo ""
    
    # 執行快速訓練
    echo -e "${BLUE}步驟 4/4: 開始快速訓練${NC}"
    echo "----------------------------------------------------------------------"
    python train_model.py \
      --meta-path datasets/meta-other.json \
      --review-path datasets/review-other.json \
      --max-pois 1000 \
      --max-reviews 5000 \
      --epochs 5 \
      --batch-size 128
    
    echo ""
    echo -e "${GREEN}✓ 快速測試完成！${NC}"
    ;;
    
  2)
    echo ""
    echo -e "${GREEN}=== 開始完整訓練 ===${NC}"
    echo ""
    echo -e "${YELLOW}注意：此過程可能需要數小時，請確保：${NC}"
    echo "  - 有足夠的磁碟空間（至少 10 GB）"
    echo "  - 有足夠的記憶體（建議 16 GB+）"
    echo "  - 電腦不會進入休眠"
    echo ""
    read -p "確定要繼續嗎？(y/N) " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
      # 安裝依賴
      echo -e "${BLUE}步驟 2/4: 安裝依賴套件${NC}"
      echo "----------------------------------------------------------------------"
      pip install -r requirements.txt
      echo ""
      
      # 測試資料載入
      echo -e "${BLUE}步驟 3/4: 測試資料載入${NC}"
      echo "----------------------------------------------------------------------"
      python -c "
from data_processor import POIDataProcessor
print('測試載入 California POI 資料...')
p = POIDataProcessor('datasets/meta-California.json.gz')
pois = p.load_data(max_records=100)
print(f'✓ 成功載入 {len(pois)} 個 POI')
"
      echo ""
      
      # 執行完整訓練
      echo -e "${BLUE}步驟 4/4: 開始完整訓練${NC}"
      echo "----------------------------------------------------------------------"
      python train_model.py \
        --meta-path datasets/meta-California.json.gz \
        --review-path datasets/review-California.json.gz \
        --max-pois 50000 \
        --max-reviews 500000 \
        --epochs 20 \
        --batch-size 256 \
        --learning-rate 0.001
      
      echo ""
      echo -e "${GREEN}✓ 完整訓練完成！${NC}"
    else
      echo "已取消"
    fi
    ;;
    
  3)
    echo ""
    echo -e "${GREEN}=== 驗證資料載入 ===${NC}"
    echo ""
    
    python -c "
from data_processor import POIDataProcessor, ReviewDataProcessor

print('測試 1: 載入測試 POI 資料')
print('-' * 60)
poi_proc = POIDataProcessor('datasets/meta-other.json')
pois = poi_proc.load_data(max_records=1000)
result = poi_proc.preprocess()
print(f'✓ 成功載入 {len(pois)} 個 POI')
print(f'✓ 類別數: {len(poi_proc.category_encoder)}')
print(f'✓ 州/城市數: {len(poi_proc.state_encoder)}')
print()

print('測試 2: 載入測試評論資料')
print('-' * 60)
review_proc = ReviewDataProcessor('datasets/review-other.json')
reviews = review_proc.load_data(max_records=5000)
result = review_proc.preprocess()
print(f'✓ 成功載入 {len(reviews)} 條評論')
print(f'✓ 用戶數: {len(review_proc.user_reviews)}')
print(f'✓ POI 評論數: {len(review_proc.poi_reviews)}')
print()

print('測試 3: 載入 California POI 資料（壓縮）')
print('-' * 60)
poi_proc_ca = POIDataProcessor('datasets/meta-California.json.gz')
pois_ca = poi_proc_ca.load_data(max_records=1000)
print(f'✓ 成功載入 {len(pois_ca)} 個 California POI')
print()

print('=' * 60)
print('✓ 所有資料載入測試通過！')
print('=' * 60)
"
    ;;
    
  4)
    echo ""
    echo -e "${GREEN}=== 資料集統計資訊 ===${NC}"
    echo ""
    
    python -c "
from data_processor import POIDataProcessor

print('分析測試資料集...')
print('=' * 70)
processor = POIDataProcessor('datasets/meta-other.json')
pois = processor.load_data(max_records=5000)
result = processor.preprocess()

print(f'''
資料集統計:
  - 總 POI 數: {len(pois)}
  - 平均評分: {processor.stats['avg_rating_mean']:.2f}
  - 評分標準差: {processor.stats['avg_rating_std']:.2f}
  - 平均評論數: {processor.stats['num_reviews_mean']:.1f}
  - 類別數: {processor.stats['num_categories']}
  - 州/城市數: {processor.stats['num_states']}

熱門類別 (Top 10):
''')
for cat, count in processor.stats['top_categories']:
    print(f'  - {cat}: {count} 個')

print()
print('=' * 70)
"
    ;;
    
  5)
    echo "再見！"
    exit 0
    ;;
    
  *)
    echo -e "${RED}無效的選項${NC}"
    exit 1
    ;;
esac

echo ""
echo "======================================================================"
echo -e "${GREEN}🎉 完成！${NC}"
echo "======================================================================"
echo ""
echo "後續步驟："
echo "  - 查看訓練日誌: cat train.log"
echo "  - 測試推薦功能: python example_usage.py"
echo "  - 查看模型檔案: ls -lh *.pth *.pkl"
echo ""
echo "相關文件："
echo "  - 資料集說明: cat datasets/README.md"
echo "  - 系統概覽: cat OVERVIEW.md"
echo "  - 使用指南: cat README.md"
echo ""
