#!/bin/bash
# 評估 DLRM 模型的推薦指標
# K = 1, 3, 10

echo "=========================================="
echo "RouteX 模型評估 - Precision, Recall, F1, NDCG, AUC"
echo "=========================================="

python3 evaluate_metrics.py \
    --model models/travel_dlrm.pth \
    --poi-data datasets/meta-California.json.gz \
    --review-data datasets/review-California.json.gz \
    --k-values 1 3 10 \
    --max-users 500 \
    --device cuda

echo ""
echo "評估完成！"
