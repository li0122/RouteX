// RouteX - 前端應用程式

// 全域變數
let map = null;
let markers = [];
let selectedCategories = [];
let routeLine = null;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initCategorySelection();
    initTopKSlider();
    initForm();
});

// 類別選擇
function initCategorySelection() {
    const categoryChips = document.querySelectorAll('.category-chip');
    
    categoryChips.forEach(chip => {
        chip.addEventListener('click', function() {
            this.classList.toggle('active');
            updateSelectedCategories();
        });
    });
}

function updateSelectedCategories() {
    selectedCategories = [];
    document.querySelectorAll('.category-chip.active').forEach(chip => {
        selectedCategories.push(chip.dataset.category);
    });
}

// Top-K 滑桿
function initTopKSlider() {
    const slider = document.getElementById('topK');
    const valueDisplay = document.getElementById('topKValue');
    
    slider.addEventListener('input', function() {
        valueDisplay.textContent = this.value;
    });
}

// 表單提交
function initForm() {
    const form = document.getElementById('recommendForm');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const startLocation = document.getElementById('startLocation').value;
        const endLocation = document.getElementById('endLocation').value;
        const topK = parseInt(document.getElementById('topK').value);
        const enableLLM = document.getElementById('enableLLM').checked;
        
        // 解析座標
        const start = parseLocation(startLocation);
        const end = parseLocation(endLocation);
        
        if (!start || !end) {
            showError('請輸入有效的座標格式 (例如: 37.7749, -122.4194)');
            return;
        }
        
        // 準備請求數據
        const requestData = {
            start_location: start,
            end_location: end,
            categories: selectedCategories,
            top_k: topK,
            enable_llm: enableLLM
        };
        
        // 發送請求
        await getRecommendations(requestData);
    });
}

// 解析地點輸入
function parseLocation(locationStr) {
    // 移除空白
    locationStr = locationStr.trim();
    
    // 嘗試解析 "緯度, 經度" 格式
    const parts = locationStr.split(',');
    if (parts.length === 2) {
        const lat = parseFloat(parts[0].trim());
        const lon = parseFloat(parts[1].trim());
        
        if (!isNaN(lat) && !isNaN(lon)) {
            return [lat, lon];
        }
    }
    
    return null;
}

// 獲取推薦
async function getRecommendations(data) {
    showLoading();
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '請求失敗');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || '發生未知錯誤，請稍後再試');
    }
}

// 顯示載入狀態
function showLoading() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('loadingState').style.display = 'block';
    
    // 禁用提交按鈕
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 處理中...';
}

// 顯示錯誤
function showError(message) {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorState').style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
    
    // 恢復提交按鈕
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-search"></i> 開始推薦';
}

// 顯示結果
function displayResults(data) {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'block';
    
    // 恢復提交按鈕
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-search"></i> 開始推薦';
    
    // 更新統計
    updateStatistics(data);
    
    // 初始化地圖
    initMap(data);
    
    // 顯示推薦列表
    displayRecommendations(data.recommendations);
}

// 更新統計資訊
function updateStatistics(data) {
    const recommendations = data.recommendations || [];
    
    document.getElementById('statTotal').textContent = recommendations.length;
    
    if (recommendations.length > 0) {
        const avgScore = recommendations.reduce((sum, rec) => sum + rec.score, 0) / recommendations.length;
        document.getElementById('statAvgScore').textContent = avgScore.toFixed(2);
        
        const totalExtraTime = recommendations.reduce((sum, rec) => sum + (rec.extra_time_minutes || 0), 0);
        document.getElementById('statExtraTime').textContent = Math.round(totalExtraTime);
    } else {
        document.getElementById('statAvgScore').textContent = '0.0';
        document.getElementById('statExtraTime').textContent = '0';
    }
}

// 初始化地圖
function initMap(data) {
    const mapDiv = document.getElementById('map');
    
    // 如果地圖已存在，先清除
    if (map) {
        map.remove();
    }
    
    // 創建新地圖
    const start = data.start_location;
    const end = data.end_location;
    const center = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
    
    map = L.map('map').setView(center, 12);
    
    // 添加地圖圖層
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // 清除舊標記
    markers = [];
    
    // 添加起點標記
    const startMarker = L.marker(start, {
        icon: L.divIcon({
            className: 'custom-marker',
            html: '<div style="background: #10b981; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><i class="fas fa-play"></i></div>',
            iconSize: [40, 40]
        })
    }).addTo(map);
    startMarker.bindPopup('<b>出發點</b>');
    markers.push(startMarker);
    
    // 添加終點標記
    const endMarker = L.marker(end, {
        icon: L.divIcon({
            className: 'custom-marker',
            html: '<div style="background: #ef4444; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><i class="fas fa-flag"></i></div>',
            iconSize: [40, 40]
        })
    }).addTo(map);
    endMarker.bindPopup('<b>目的地</b>');
    markers.push(endMarker);
    
    // 添加推薦點標記
    data.recommendations.forEach((rec, index) => {
        const poi = rec.poi;
        const marker = L.marker([poi.latitude, poi.longitude], {
            icon: L.divIcon({
                className: 'custom-marker',
                html: `<div style="background: #2563eb; color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">${index + 1}</div>`,
                iconSize: [35, 35]
            })
        }).addTo(map);
        
        marker.bindPopup(`
            <div style="min-width: 200px;">
                <h4 style="margin: 0 0 10px 0; color: #2563eb;">${poi.name}</h4>
                <p style="margin: 5px 0;"><strong>類別:</strong> ${poi.primary_category || '未分類'}</p>
                <p style="margin: 5px 0;"><strong>評分:</strong> ${poi.avg_rating ? poi.avg_rating.toFixed(1) : 'N/A'} ⭐</p>
                <p style="margin: 5px 0;"><strong>AI評分:</strong> ${rec.score.toFixed(3)}</p>
                <p style="margin: 5px 0;"><strong>額外時間:</strong> ${rec.extra_time_minutes ? rec.extra_time_minutes.toFixed(1) : 'N/A'} 分鐘</p>
            </div>
        `);
        
        markers.push(marker);
    });
    
    // 畫路線
    if (routeLine) {
        map.removeLayer(routeLine);
    }
    
    const routePoints = [start];
    data.recommendations.forEach(rec => {
        routePoints.push([rec.poi.latitude, rec.poi.longitude]);
    });
    routePoints.push(end);
    
    routeLine = L.polyline(routePoints, {
        color: '#2563eb',
        weight: 3,
        opacity: 0.7,
        dashArray: '10, 10'
    }).addTo(map);
    
    // 調整視角以顯示所有標記
    const bounds = L.latLngBounds(routePoints);
    map.fitBounds(bounds, { padding: [50, 50] });
}

// 顯示推薦列表
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsList');
    container.innerHTML = '';
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #9ca3af; padding: 40px;">沒有找到符合條件的推薦</p>';
        return;
    }
    
    recommendations.forEach((rec, index) => {
        const card = createRecommendationCard(rec, index + 1);
        container.appendChild(card);
    });
}

// 創建推薦卡片
function createRecommendationCard(rec, rank) {
    const poi = rec.poi;
    const card = document.createElement('div');
    card.className = 'recommendation-card';
    card.style.animationDelay = `${rank * 0.1}s`;
    
    // 構建卡片HTML
    card.innerHTML = `
        <div class="card-header">
            <div class="card-rank">${rank}</div>
            <div class="card-title">
                <h3>${poi.name || '未知地點'}</h3>
                <span class="card-category">
                    <i class="fas fa-tag"></i> ${poi.primary_category || '未分類'}
                </span>
            </div>
            <div class="card-score">
                ${rec.score.toFixed(3)}
            </div>
        </div>
        
        <div class="card-body">
            <div class="card-info">
                <div class="info-item">
                    <i class="fas fa-star"></i>
                    <span>${poi.avg_rating ? poi.avg_rating.toFixed(1) : 'N/A'} ⭐</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-comment"></i>
                    <span>${poi.num_reviews || 0} 評論</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-clock"></i>
                    <span>+${rec.extra_time_minutes ? rec.extra_time_minutes.toFixed(1) : 'N/A'} 分鐘</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-map-pin"></i>
                    <span>${poi.latitude.toFixed(4)}, ${poi.longitude.toFixed(4)}</span>
                </div>
            </div>
            
            ${rec.llm_approved ? '<span class="llm-badge"><i class="fas fa-check-circle"></i> AI審核通過</span>' : ''}
            
            <div class="card-details">
                <div class="detail-row">
                    <span class="detail-label">繞道比例</span>
                    <span class="detail-value">${rec.detour_info ? (rec.detour_info.detour_ratio || 1).toFixed(2) : 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">額外距離</span>
                    <span class="detail-value">${rec.detour_info && rec.detour_info.extra_duration ? (rec.detour_info.extra_duration / 60).toFixed(1) : 'N/A'} 分鐘</span>
                </div>
                ${rec.reasons && rec.reasons.length > 0 ? `
                <div class="detail-row">
                    <span class="detail-label">推薦理由</span>
                    <span class="detail-value">${rec.reasons.slice(0, 2).join(', ')}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
    
    // 點擊卡片時在地圖上高亮顯示
    card.addEventListener('click', function() {
        if (markers[rank + 1]) {  // +1 因為前兩個是起點和終點
            markers[rank + 1].openPopup();
            map.setView([poi.latitude, poi.longitude], 15);
        }
    });
    
    return card;
}

// 工具函數：格式化數字
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}
