// RouteX - 前端應用程式

// 全域變數
let map = null;
let markers = [];
let selectedCategories = [];
let routeLine = null;
let offlineMapPicker = null;
let startLocation = null;
let endLocation = null;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initCategorySelection();
    initTopKSlider();
    initOfflineMapPicker();
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

// 離線地圖選擇器
function initOfflineMapPicker() {
    offlineMapPicker = new OfflineMapPicker('locationPickerMap', {
        bounds: {
            minLat: 37.6,
            maxLat: 37.9,
            minLng: -122.6,
            maxLng: -122.2
        },
        onStartSelect: (lat, lng) => {
            startLocation = [lat, lng];
            document.getElementById('startLocation').value = `${lat},${lng}`;
            document.getElementById('startLocationDisplay').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            document.getElementById('setStartBtn').style.opacity = '0.7';
            document.getElementById('setStartBtn').style.transform = 'scale(1)';
            showToast('出發點已設定！', 'success');
        },
        onEndSelect: (lat, lng) => {
            endLocation = [lat, lng];
            document.getElementById('endLocation').value = `${lat},${lng}`;
            document.getElementById('endLocationDisplay').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            document.getElementById('setEndBtn').style.opacity = '0.7';
            document.getElementById('setEndBtn').style.transform = 'scale(1)';
            showToast('目的地已設定！', 'success');
        }
    });
    
    // 設定出發點按鈕
    document.getElementById('setStartBtn').addEventListener('click', function() {
        offlineMapPicker.setSelectingStart(true);
        this.style.opacity = '1';
        this.style.transform = 'scale(1.05)';
        document.getElementById('setEndBtn').style.opacity = '0.7';
        document.getElementById('setEndBtn').style.transform = 'scale(1)';
        showToast('請在地圖上點擊選擇出發點', 'info');
    });
    
    // 設定目的地按鈕
    document.getElementById('setEndBtn').addEventListener('click', function() {
        offlineMapPicker.setSelectingEnd(true);
        this.style.opacity = '1';
        this.style.transform = 'scale(1.05)';
        document.getElementById('setStartBtn').style.opacity = '0.7';
        document.getElementById('setStartBtn').style.transform = 'scale(1)';
        showToast('請在地圖上點擊選擇目的地', 'info');
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
        
        // 獲取活動意圖
        const activityIntent = document.getElementById('activityIntent').value.trim();
        
        const topK = parseInt(document.getElementById('topK').value);
        const enableLLM = document.getElementById('enableLLM').checked;
        
        // 準備請求數據
        const requestData = {
            start_location: startLocation,
            end_location: endLocation,
            activity_intent: activityIntent,  // 使用活動意圖代替類別
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
    // 創建離線結果地圖
    if (!map) {
        map = new OfflineResultMap('map');
    }
    
    // 設置數據並繪製
    map.setData(data.start_location, data.end_location, data.recommendations);
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
            </div>
            
            <div class="card-location">
                <div class="location-coords">
                    <i class="fas fa-map-marker-alt"></i>
                    <span class="coord-label">座標:</span>
                    <span class="coord-value" title="點擊複製座標">
                        <span class="lat">${poi.latitude.toFixed(6)}</span>, 
                        <span class="lng">${poi.longitude.toFixed(6)}</span>
                    </span>
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
    
    // 添加點擊複製座標功能
    const coordValue = card.querySelector('.coord-value');
    if (coordValue) {
        coordValue.addEventListener('click', function() {
            const coordText = `${poi.latitude}, ${poi.longitude}`;
            navigator.clipboard.writeText(coordText).then(() => {
                // 顯示複製成功提示
                const originalTitle = coordValue.getAttribute('title');
                coordValue.setAttribute('title', '已複製! ✓');
                coordValue.style.background = 'rgba(34, 197, 94, 0.1)';
                
                setTimeout(() => {
                    coordValue.setAttribute('title', originalTitle);
                    coordValue.style.background = '';
                }, 1500);
            }).catch(err => {
                console.error('複製失敗:', err);
            });
        });
    }
    
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
