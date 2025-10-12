// RouteX - 前端應用程式

// 全域變數
let map = null;
let markers = [];
let selectedCategories = [];
let routeLine = null;
let locationPickerMap = null;
let startLocation = null;
let endLocation = null;
let isSelectingStart = false;
let isSelectingEnd = false;
let startMarker = null;
let endMarker = null;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initCategorySelection();
    initTopKSlider();
    initLocationPicker();
    initForm();
});

// 類別選擇
function initCategorySelection() {
    const categoryChips = document.querySelectorAll('.category-chip');
    console.log('初始化類別選擇，找到', categoryChips.length, '個類別');
    
    categoryChips.forEach(chip => {
        chip.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.toggle('active');
            updateSelectedCategories();
            console.log('點擊類別:', chip.dataset.category, '當前選中:', selectedCategories);
        });
        
        // 確保可點擊樣式
        chip.style.cursor = 'pointer';
        chip.style.userSelect = 'none';
    });
}

function updateSelectedCategories() {
    selectedCategories = [];
    document.querySelectorAll('.category-chip.active').forEach(chip => {
        selectedCategories.push(chip.dataset.category);
    });
}

// 地點選擇器
function initLocationPicker() {
    // 初始化地圖（預設中心：舊金山）
    locationPickerMap = L.map('locationPickerMap').setView([37.7749, -122.4194], 11);
    
    // 添加地圖圖層
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(locationPickerMap);
    
    // 設定出發點按鈕
    document.getElementById('setStartBtn').addEventListener('click', function() {
        isSelectingStart = true;
        isSelectingEnd = false;
        this.style.opacity = '1';
        this.style.transform = 'scale(1.05)';
        document.getElementById('setEndBtn').style.opacity = '0.7';
        document.getElementById('setEndBtn').style.transform = 'scale(1)';
        showToast('請在地圖上點擊選擇出發點', 'info');
    });
    
    // 設定目的地按鈕
    document.getElementById('setEndBtn').addEventListener('click', function() {
        isSelectingEnd = true;
        isSelectingStart = false;
        this.style.opacity = '1';
        this.style.transform = 'scale(1.05)';
        document.getElementById('setStartBtn').style.opacity = '0.7';
        document.getElementById('setStartBtn').style.transform = 'scale(1)';
        showToast('請在地圖上點擊選擇目的地', 'info');
    });
    
    // 地圖點擊事件
    locationPickerMap.on('click', function(e) {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;
        
        if (isSelectingStart) {
            // 設定出發點
            startLocation = [lat, lng];
            document.getElementById('startLocation').value = `${lat},${lng}`;
            document.getElementById('startLocationDisplay').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            
            // 移除舊標記
            if (startMarker) {
                locationPickerMap.removeLayer(startMarker);
            }
            
            // 添加新標記
            startMarker = L.marker([lat, lng], {
                icon: L.divIcon({
                    className: 'custom-marker',
                    html: '<div style="background: #10b981; color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><i class="fas fa-play"></i></div>',
                    iconSize: [35, 35]
                })
            }).addTo(locationPickerMap);
            startMarker.bindPopup('<b>出發點</b>').openPopup();
            
            isSelectingStart = false;
            document.getElementById('setStartBtn').style.opacity = '0.7';
            document.getElementById('setStartBtn').style.transform = 'scale(1)';
            showToast('出發點已設定！', 'success');
            
        } else if (isSelectingEnd) {
            // 設定目的地
            endLocation = [lat, lng];
            document.getElementById('endLocation').value = `${lat},${lng}`;
            document.getElementById('endLocationDisplay').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            
            // 移除舊標記
            if (endMarker) {
                locationPickerMap.removeLayer(endMarker);
            }
            
            // 添加新標記
            endMarker = L.marker([lat, lng], {
                icon: L.divIcon({
                    className: 'custom-marker',
                    html: '<div style="background: #ef4444; color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><i class="fas fa-flag"></i></div>',
                    iconSize: [35, 35]
                })
            }).addTo(locationPickerMap);
            endMarker.bindPopup('<b>目的地</b>').openPopup();
            
            isSelectingEnd = false;
            document.getElementById('setEndBtn').style.opacity = '0.7';
            document.getElementById('setEndBtn').style.transform = 'scale(1)';
            showToast('目的地已設定！', 'success');
        }
        
        // 如果兩個點都設定了，調整視角
        if (startLocation && endLocation) {
            const bounds = L.latLngBounds([startLocation, endLocation]);
            locationPickerMap.fitBounds(bounds, { padding: [50, 50] });
        }
    });
}

// 顯示提示訊息
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#2563eb'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
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
        
        // 驗證地點是否已選擇
        if (!startLocation || !endLocation) {
            showError('請先在地圖上選擇出發點和目的地！');
            return;
        }
        
        const topK = parseInt(document.getElementById('topK').value);
        const enableLLM = document.getElementById('enableLLM').checked;
        
        // 準備請求數據
        const requestData = {
            start_location: startLocation,
            end_location: endLocation,
            categories: selectedCategories,
            top_k: topK,
            enable_llm: enableLLM
        };
        
        console.log('提交推薦請求:', requestData);
        
        // 發送請求
        await getRecommendations(requestData);
    });
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
