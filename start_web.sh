#!/bin/bash

# RouteX Web Application Startup Script

echo "======================================"
echo "🚀 RouteX Web Application"
echo "======================================"
echo ""

# 檢查Python環境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安裝"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"
echo ""

# 檢查必要的文件
if [ ! -f "datasets/meta-California.json.gz" ]; then
    echo "⚠️ 警告: POI數據集未找到 (datasets/meta-California.json.gz)"
fi

if [ ! -f "models/travel_dlrm.pth" ]; then
    echo "⚠️ 警告: 模型文件未找到 (models/travel_dlrm.pth)"
fi

echo ""
echo "📦 檢查依賴..."

# 檢查Flask是否安裝
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Flask 未安裝"
    echo "正在安裝依賴..."
    pip install -r requirements_web.txt
else
    echo "✅ Flask 已安裝"
fi

echo ""
echo "======================================"
echo "🌐 啟動 Web 服務器..."
echo "======================================"
echo ""
echo "訪問地址: http://localhost:5000"
echo "按 Ctrl+C 停止服務器"
echo ""

# 啟動Flask應用
python3 web_app.py
