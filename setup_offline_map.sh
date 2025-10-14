#!/bin/bash
#
# 離線地圖快速設置腳本
# 自動下載舊金山灣區瓦片包
#

set -e

echo "============================================================"
echo "🗺️  RouteX 離線地圖設置"
echo "============================================================"
echo ""

# 配置
BBOX="37.6,-122.6,37.9,-122.2"
ZOOM="8-13"
OUTPUT="static/data/sf_bay_area.mbtiles"
WORKERS=8

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安裝"
    exit 1
fi

# 檢查必要的 Python 庫
echo "📦 檢查依賴..."
python3 -c "import requests, sqlite3" 2>/dev/null || {
    echo "❌ 缺少必要的 Python 庫"
    echo "   請執行: pip3 install requests"
    exit 1
}

# 創建數據目錄
echo "📁 創建數據目錄..."
mkdir -p static/data

# 檢查是否已存在
if [ -f "$OUTPUT" ]; then
    echo ""
    echo "⚠️  檔案已存在: $OUTPUT"
    read -p "是否覆蓋？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消下載"
        exit 0
    fi
    rm -f "$OUTPUT"
fi

# 顯示配置
echo ""
echo "📋 下載配置："
echo "   區域: 舊金山灣區"
echo "   邊界: $BBOX"
echo "   縮放級別: $ZOOM"
echo "   輸出檔案: $OUTPUT"
echo "   並發數: $WORKERS"
echo ""

# 估算
echo "⏱️  預計下載時間: 10-20 分鐘"
echo "💾 預計檔案大小: 150-300 MB"
echo ""

read -p "開始下載？(Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "已取消"
    exit 0
fi

# 開始下載
echo ""
echo "🚀 開始下載瓦片..."
echo "============================================================"
echo ""

python3 download_tiles.py \
    --bbox "$BBOX" \
    --zoom "$ZOOM" \
    --output "$OUTPUT" \
    --workers "$WORKERS"

# 檢查結果
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "============================================================"
    echo "✅ 下載完成！"
    echo "   檔案: $OUTPUT"
    echo "   大小: $SIZE"
    echo "============================================================"
    echo ""
    echo "📝 下一步："
    echo "   1. 編輯 static/js/app.js"
    echo "   2. 設置 useMBTiles: true"
    echo "   3. 啟動伺服器: python3 test_leaflet_simple.py"
    echo "   4. 訪問: http://localhost:5050/test_leaflet"
    echo ""
    echo "📖 詳細說明請參考: QUICKSTART_OFFLINE.md"
    echo ""
else
    echo ""
    echo "❌ 下載失敗，請檢查錯誤訊息"
    exit 1
fi
