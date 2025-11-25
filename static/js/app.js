// RouteX - 前端應用程式

// 全域變數
let map = null;
let mapPicker = null;  // 改用 Leaflet 地圖選擇器
let markers = [];
let selectedCategories = [];
let routeLine = null;
let startLocation = null;
let endLocation = null;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initStepNavigation();    // 初始化步驟導航
    initProfileSettings();   // 初始化用戶畫像設定
    initModeSelector();
    initCategorySelection();
    initTopKSlider();
    initTimeBudgetSlider();
    initLeafletMapPicker();  // 使用 Leaflet
    initForm();
    loadCategories();        // 載入類別列表
});

// 初始化步驟導航
function initStepNavigation() {
    const nextStepBtn = document.getElementById('nextStepBtn');
    const backToStep1Btn = document.getElementById('backToStep1Btn');
    const step1Section = document.getElementById('step1Section');
    const step2Section = document.getElementById('step2Section');
    
    // 下一步按鈕
    nextStepBtn.addEventListener('click', function() {
        // 平滑滾動到頂部
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // 切換步驟顯示
        step1Section.style.display = 'none';
        step2Section.style.display = 'block';
        
        // 顯示畫像摘要提示
        showProfileSummary();
        
        showToast('✅ 畫像已設定！現在選擇推薦模式', 'success');
    });
    
    // 返回按鈕
    backToStep1Btn.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        step1Section.style.display = 'block';
        step2Section.style.display = 'none';
        showToast('返回畫像設定', 'info');
    });
}

// 顯示畫像摘要
function showProfileSummary() {
    const avgRating = document.getElementById('avgRating').value;
    const budget = document.getElementById('budget').value;
    const categoryCheckboxes = document.querySelectorAll('.category-checkbox:checked');
    const selectedCount = categoryCheckboxes.length;
    
    const budgetLabels = ['', '節省', '經濟', '中等', '舒適', '奢華'];
    
    console.log(`👤 用戶畫像: 評分標準 ${avgRating}⭐ | 預算 ${budgetLabels[budget]} | ${selectedCount} 個類別偏好`);
}

// 模式選擇器
function initModeSelector() {
    const modeOptions = document.querySelectorAll('input[name="recommendMode"]');
    
    modeOptions.forEach(option => {
        option.addEventListener('change', function() {
            updateUIForMode(this.value);
        });
    });
    
    // 初始化為單點推薦模式
    updateUIForMode('poi');
}

function updateUIForMode(mode) {
    const activityIntentGroup = document.getElementById('activityIntentGroup');
    const locationPickerGroup = document.getElementById('locationPickerGroup');
    const topKGroup = document.getElementById('topKGroup');
    const timeBudgetGroup = document.getElementById('timeBudgetGroup');
    const llmToggleGroup = document.getElementById('llmToggleGroup');
    const submitBtnText = document.getElementById('submitBtnText');
    const poiHints = document.querySelectorAll('.poi-hint');
    const itineraryHints = document.querySelectorAll('.itinerary-hint');
    const topKSlider = document.getElementById('topK');
    
    // 隱藏所有提示
    poiHints.forEach(hint => hint.style.display = 'none');
    itineraryHints.forEach(hint => hint.style.display = 'none');
    
    if (mode === 'poi') {
        // 路線推薦模式
        activityIntentGroup.style.display = 'block';
        locationPickerGroup.style.display = 'block';
        timeBudgetGroup.style.display = 'none';
        llmToggleGroup.style.display = 'block';
        
        topKSlider.min = '3';
        topKSlider.max = '10';
        topKSlider.value = '5';
        document.getElementById('topKValue').textContent = '5';
        
        submitBtnText.textContent = '生成路線推薦';
        poiHints.forEach(hint => hint.style.display = 'block');
        
    } else if (mode === 'itinerary') {
        // 行程規劃模式
        activityIntentGroup.style.display = 'block';
        locationPickerGroup.style.display = 'block';
        timeBudgetGroup.style.display = 'block';
        llmToggleGroup.style.display = 'none';
        
        topKSlider.min = '10';
        topKSlider.max = '30';
        topKSlider.value = '20';
        document.getElementById('topKValue').textContent = '20';
        
        submitBtnText.textContent = '生成完整行程';
        itineraryHints.forEach(hint => hint.style.display = 'block');
    }
}

// 時間預算滑桿
function initTimeBudgetSlider() {
    const slider = document.getElementById('timeBudget');
    const valueDisplay = document.getElementById('timeBudgetValue');
    
    if (slider && valueDisplay) {
        slider.addEventListener('input', function() {
            valueDisplay.textContent = this.value;
        });
    }
}

// 初始化用戶畫像設定
function initProfileSettings() {
    // 評分習慣滑桿
    const avgRatingSlider = document.getElementById('avgRating');
    const avgRatingValue = document.getElementById('avgRatingValue');
    if (avgRatingSlider && avgRatingValue) {
        avgRatingSlider.addEventListener('input', function() {
            avgRatingValue.textContent = this.value;
        });
    }
    
    // 評分寬容度滑桿
    const ratingStdSlider = document.getElementById('ratingStd');
    const ratingStdValue = document.getElementById('ratingStdValue');
    if (ratingStdSlider && ratingStdValue) {
        ratingStdSlider.addEventListener('input', function() {
            ratingStdValue.textContent = this.value;
        });
    }
    
    // 評論數量滑桿
    const numReviewsSlider = document.getElementById('numReviews');
    const numReviewsValue = document.getElementById('numReviewsValue');
    if (numReviewsSlider && numReviewsValue) {
        numReviewsSlider.addEventListener('input', function() {
            numReviewsValue.textContent = this.value;
        });
    }
    
    // 預算等級滑桿
    const budgetSlider = document.getElementById('budget');
    const budgetValue = document.getElementById('budgetValue');
    if (budgetSlider && budgetValue) {
        budgetSlider.addEventListener('input', function() {
            budgetValue.textContent = this.value;
        });
    }
    
    // 價格範圍滑桿
    const priceMinSlider = document.getElementById('priceMin');
    const priceMaxSlider = document.getElementById('priceMax');
    const priceRangeDisplay = document.getElementById('priceRangeDisplay');
    
    function updatePriceRange() {
        const min = parseInt(priceMinSlider.value);
        const max = parseInt(priceMaxSlider.value);
        
        // 確保 min <= max
        if (min > max) {
            priceMinSlider.value = max;
        }
        
        const priceLabels = ['$', '$$', '$$$', '$$$$', '$$$$$'];
        priceRangeDisplay.textContent = `${priceLabels[min]} - ${priceLabels[max]}`;
    }
    
    if (priceMinSlider && priceMaxSlider) {
        priceMinSlider.addEventListener('input', updatePriceRange);
        priceMaxSlider.addEventListener('input', updatePriceRange);
        updatePriceRange();
    }
}

// 載入類別列表
async function loadCategories() {
    try {
        const response = await fetch('/api/categories');
        const data = await response.json();
        
        // 填充類別複選框
        const categoryCheckboxes = document.getElementById('categoryCheckboxes');
        if (categoryCheckboxes && data.categories) {
            categoryCheckboxes.innerHTML = '';
            
            // 添加「全選」選項
            const allCheckbox = document.createElement('label');
            allCheckbox.style.cssText = 'display: flex; align-items: center; gap: 6px; padding: 8px; background: #f3f4f6; border-radius: 6px; cursor: pointer; user-select: none;';
            allCheckbox.innerHTML = `
                <input type="checkbox" value="all" style="cursor: pointer;" checked>
                <span style="font-weight: 500;">全選</span>
            `;
            categoryCheckboxes.appendChild(allCheckbox);
            
            // 添加各類別
            data.categories.slice(0, 20).forEach(category => {
                const label = document.createElement('label');
                label.style.cssText = 'display: flex; align-items: center; gap: 6px; padding: 8px; background: #f9fafb; border-radius: 6px; cursor: pointer; user-select: none;';
                label.innerHTML = `
                    <input type="checkbox" value="${category}" class="category-checkbox" style="cursor: pointer;" checked>
                    <span style="font-size: 14px;">${category}</span>
                `;
                categoryCheckboxes.appendChild(label);
            });
            
            // 全選邏輯
            const allCheckboxInput = allCheckbox.querySelector('input');
            const categoryCheckboxInputs = categoryCheckboxes.querySelectorAll('.category-checkbox');
            
            allCheckboxInput.addEventListener('change', function() {
                categoryCheckboxInputs.forEach(cb => cb.checked = this.checked);
            });
            
            categoryCheckboxInputs.forEach(cb => {
                cb.addEventListener('change', function() {
                    const allChecked = Array.from(categoryCheckboxInputs).every(c => c.checked);
                    allCheckboxInput.checked = allChecked;
                });
            });
        }
        
        // 填充州列表
        const stateFilter = document.getElementById('stateFilter');
        if (stateFilter && data.states) {
            data.states.forEach(state => {
                const option = document.createElement('option');
                option.value = state;
                option.textContent = state;
                stateFilter.appendChild(option);
            });
        }
    } catch (error) {
        console.error('載入類別失敗:', error);
    }
}

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

// Leaflet 地圖選擇器
function initLeafletMapPicker() {
    mapPicker = new LeafletMapPicker('locationPickerMap');
    
    // 設定出發點按鈕
    document.getElementById('setStartBtn').addEventListener('click', function() {
        mapPicker.setMode('start');
        this.style.opacity = '1';
        this.style.transform = 'scale(1.05)';
        document.getElementById('setEndBtn').style.opacity = '0.7';
        document.getElementById('setEndBtn').style.transform = 'scale(1)';
        showToast('請在地圖上點擊選擇出發點', 'info');
    });
    
    // 設定目的地按鈕
    document.getElementById('setEndBtn').addEventListener('click', function() {
        mapPicker.setMode('end');
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
        
        // 獲取用戶畫像（步驟1）
        const userProfile = {
            avg_rating: parseFloat(document.getElementById('avgRating').value),
            rating_std: parseFloat(document.getElementById('ratingStd').value),
            num_reviews: parseInt(document.getElementById('numReviews').value),
            budget: parseInt(document.getElementById('budget').value)
        };
        
        // 獲取類別偏好
        const categoryCheckboxes = document.querySelectorAll('.category-checkbox:checked');
        const selectedCategories = Array.from(categoryCheckboxes).map(cb => cb.value);
        
        // 獲取地區和價格範圍
        const state = document.getElementById('stateFilter').value;
        const priceMin = parseInt(document.getElementById('priceMin').value);
        const priceMax = parseInt(document.getElementById('priceMax').value);
        
        const filters = {
            categories: selectedCategories,
            state: state,
            price_range: [priceMin, priceMax]
        };
        
        // 獲取選擇的模式（步驟2）
        const mode = document.querySelector('input[name="recommendMode"]:checked').value;
        
        // 準備請求數據
        let requestData, apiEndpoint;
        
        // 從地圖選擇器獲取起終點
        const start = mapPicker.startLocation;
        const end = mapPicker.endLocation;
        
        // 驗證地點是否已選擇
        if (!start || !end) {
            showError('請先在地圖上選擇出發點和目的地！');
            return;
        }
        
        // 獲取活動意圖
        const activityIntent = document.getElementById('activityIntent').value.trim();
        const topK = parseInt(document.getElementById('topK').value);
        
        // 構建用戶歷史（基於畫像）
        const userHistory = selectedCategories.map(category => ({
            category: category,
            rating: userProfile.avg_rating
        }));
        
        if (mode === 'poi') {
            // 路線推薦（帶用戶畫像）
            const enableLLM = document.getElementById('enableLLM').checked;
            requestData = {
                start_location: start,
                end_location: end,
                activity_intent: activityIntent,
                user_profile: userProfile,
                user_history: userHistory,
                top_k: topK,
                enable_llm: enableLLM
            };
            apiEndpoint = '/api/recommend';
            
        } else if (mode === 'itinerary') {
            // 行程推薦（帶用戶畫像）
            const timeBudget = parseInt(document.getElementById('timeBudget').value);
            requestData = {
                start: start,
                end: end,
                activity_intent: activityIntent || '旅遊探索',
                time_budget: timeBudget,
                user_profile: userProfile,
                user_history: userHistory,
                top_k: topK,
                user_id: 'web_user'
            };
            apiEndpoint = '/api/itinerary';
        }
        
        console.log('提交推薦請求:', { 
            mode, 
            endpoint: apiEndpoint, 
            userProfile,
            filters,
            data: requestData 
        });
        
        // 發送請求
        await getRecommendations(requestData, apiEndpoint, mode);
    });
}

// 獲取推薦
async function getRecommendations(data, endpoint, mode) {
    showLoading();
    
    try {
        const response = await fetch(endpoint, {
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
        displayResults(result, mode);
        
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
function displayResults(data, mode) {
    console.log('顯示結果:', data, 'mode:', mode);
    
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'block';
    
    // 恢復提交按鈕
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = false;
    let submitBtnText = '生成路線推薦';
    if (mode === 'itinerary') submitBtnText = '生成完整行程';
    submitBtn.innerHTML = `<i class="fas fa-magic"></i><span id="submitBtnText"> ${submitBtnText}</span>`;
    
    if (mode === 'itinerary' && data.type === 'itinerary') {
        // 行程推薦模式
        displayItineraryResult(data);
    } else {
        // 路線推薦模式
        displayPOIResults(data);
    }
}

// 顯示單點推薦結果
function displayPOIResults(data) {
    // 更新統計
    updateStatistics(data);
    
    // 初始化地圖
    initMap(data);
    
    // 顯示推薦列表
    displayRecommendations(data.recommendations);
}

// 顯示用戶畫像推薦結果
function displayProfileResults(data) {
    // 更新統計為畫像統計
    const recommendations = data.recommendations || [];
    
    document.getElementById('statTotal').textContent = recommendations.length;
    
    if (recommendations.length > 0) {
        const avgScore = recommendations.reduce((sum, rec) => sum + rec.score, 0) / recommendations.length;
        document.getElementById('statAvgScore').textContent = avgScore.toFixed(3);
        
        const avgRating = recommendations.reduce((sum, rec) => sum + (rec.rating || 0), 0) / recommendations.length;
        document.getElementById('statExtraTime').textContent = avgRating.toFixed(1);
    } else {
        document.getElementById('statAvgScore').textContent = '0.0';
        document.getElementById('statExtraTime').textContent = '0.0';
    }
    
    // 更新統計標籤
    document.querySelector('.stats-container .stat-card:nth-child(1) .stat-label').textContent = '推薦景點';
    document.querySelector('.stats-container .stat-card:nth-child(2) .stat-label').textContent = '平均分數';
    document.querySelector('.stats-container .stat-card:nth-child(3) .stat-label').textContent = '平均評分';
    
    // 初始化地圖（用戶畫像模式）
    initProfileMap(recommendations);
    
    // 顯示推薦卡片
    displayProfileRecommendations(recommendations);
}

// 初始化用戶畫像地圖
function initProfileMap(recommendations) {
    if (!map) {
        map = new LeafletResultMap('map');
    }
    
    // 構建地圖數據（不需要路線）
    const mapData = {
        recommendations: recommendations.map((rec, idx) => ({
            poi: {
                name: rec.name,
                latitude: rec.latitude,
                longitude: rec.longitude,
                avg_rating: rec.rating,
                num_reviews: rec.num_reviews,
                primary_category: rec.category
            },
            score: rec.score
        }))
    };
    
    window.map = map;
    map.setData(mapData, 'profile'); // 'profile' 模式不顯示路線
}

// 顯示用戶畫像推薦卡片
function displayProfileRecommendations(recommendations) {
    const container = document.getElementById('recommendationsList');
    container.innerHTML = '';
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #9ca3af; padding: 40px;">沒有找到符合條件的推薦</p>';
        return;
    }
    
    recommendations.forEach((rec, index) => {
        const card = createProfileRecommendationCard(rec, index + 1);
        container.appendChild(card);
    });
}

// 創建用戶畫像推薦卡片
function createProfileRecommendationCard(rec, rank) {
    const card = document.createElement('div');
    card.className = 'recommendation-card';
    card.style.animationDelay = `${rank * 0.05}s`;
    
    const priceLabels = ['免費', '$', '$$', '$$$', '$$$$', '$$$$$'];
    const priceLevel = rec.price_level || 0;
    
    card.innerHTML = `
        <div class="card-header">
            <div class="card-rank">${rank}</div>
            <div class="card-title">
                <h3>${rec.name || '未知地點'}</h3>
                <span class="card-category">
                    <i class="fas fa-tag"></i> ${rec.category || '未分類'}
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
                    <span>${rec.rating ? rec.rating.toFixed(1) : 'N/A'} ⭐</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-comment"></i>
                    <span>${rec.num_reviews ? rec.num_reviews.toLocaleString() : '0'} 評論</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-dollar-sign"></i>
                    <span>${priceLabels[priceLevel]}</span>
                </div>
            </div>
            
            <div class="card-location">
                <div class="location-coords">
                    <i class="fas fa-map-marker-alt"></i>
                    <span class="coord-label">座標:</span>
                    <span class="coord-value" title="點擊複製座標">
                        <span class="lat">${rec.latitude ? rec.latitude.toFixed(6) : 'N/A'}</span>, 
                        <span class="lng">${rec.longitude ? rec.longitude.toFixed(6) : 'N/A'}</span>
                    </span>
                </div>
                ${rec.state ? `
                <div style="margin-top: 8px;">
                    <i class="fas fa-map"></i>
                    <span style="color: #6b7280;">${rec.state}</span>
                </div>
                ` : ''}
                ${rec.address ? `
                <div style="margin-top: 8px; font-size: 13px; color: #6b7280;">
                    <i class="fas fa-location-dot"></i> ${rec.address}
                </div>
                ` : ''}
            </div>
        </div>
    `;
    
    // 添加點擊標記功能
    card.addEventListener('click', function(e) {
        if (e.target.closest('.coord-value')) {
            return;
        }
        
        if (window.map && rec.latitude && rec.longitude) {
            window.map.map.setView([rec.latitude, rec.longitude], 14);
            
            // 高亮當前卡片
            document.querySelectorAll('.recommendation-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        }
    });
    
    card.style.cursor = 'pointer';
    
    // 添加點擊複製座標功能
    const coordValue = card.querySelector('.coord-value');
    if (coordValue) {
        coordValue.style.cursor = 'pointer';
        coordValue.addEventListener('click', function(e) {
            e.stopPropagation();
            const coordText = `${rec.latitude}, ${rec.longitude}`;
            navigator.clipboard.writeText(coordText).then(() => {
                const originalTitle = coordValue.getAttribute('title');
                coordValue.setAttribute('title', '✅ 已複製!');
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

// 顯示行程推薦結果
function displayItineraryResult(data) {
    const itinerary = data.itinerary;
    
    // 更新統計為行程統計
    document.getElementById('statTotal').textContent = itinerary.total_stops;
    document.getElementById('statAvgScore').textContent = `${itinerary.total_duration}分`;
    document.getElementById('statExtraTime').textContent = `${itinerary.total_distance.toFixed(1)}km`;
    
    // 更新統計標籤
    document.querySelector('.stats-container .stat-card:nth-child(1) .stat-label').textContent = '景點數量';
    document.querySelector('.stats-container .stat-card:nth-child(2) .stat-label').textContent = '預計時間';
    document.querySelector('.stats-container .stat-card:nth-child(3) .stat-label').textContent = '總距離';
    
    // 初始化地圖（行程模式）
    initItineraryMap(itinerary);
    
    // 顯示行程卡片
    displayItineraryCard(itinerary);
}

// 初始化行程地圖
function initItineraryMap(itinerary) {
    if (!map) {
        map = new LeafletResultMap('map');
    }
    
    // 驗證並過濾有效的景點數據
    const validStops = itinerary.stops.filter(stop => {
        return stop && 
               typeof stop.latitude === 'number' && 
               typeof stop.longitude === 'number' &&
               !isNaN(stop.latitude) && 
               !isNaN(stop.longitude);
    });
    
    if (validStops.length === 0) {
        console.error(' 沒有有效的景點數據');
        return;
    }
    
    console.log(' 有效景點:', validStops.length, '/', itinerary.stops.length);
    
    // 構建行程路線數據
    const routeData = {
        start_location: itinerary.route.start,
        end_location: itinerary.route.end,
        recommendations: validStops.map((stop, idx) => ({
            poi: {
                name: stop.name || 'Unknown',
                latitude: stop.latitude,
                longitude: stop.longitude,
                avg_rating: stop.rating || 0,
                num_reviews: stop.reviews || 0,
                primary_category: stop.category || 'N/A'
            },
            score: 1.0 - (idx * 0.1),
            extra_time_minutes: stop.duration || 60
        }))
    };
    
    console.log(' 路線數據:', routeData);
    
    // 保存到全局變量供卡片點擊使用
    window.map = map;
    
    try {
        map.setData(routeData, 'itinerary'); // 傳遞 'itinerary' 模式
    } catch (error) {
        console.error(' 地圖初始化錯誤:', error);
    }
}

// 顯示行程卡片
function displayItineraryCard(itinerary) {
    const container = document.getElementById('recommendationsList');
    container.innerHTML = '';
    
    const card = document.createElement('div');
    card.className = 'itinerary-card';
    
    // 驗證數據
    if (!itinerary || !itinerary.stops || itinerary.stops.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #9ca3af; padding: 40px;">無法生成行程</p>';
        return;
    }
    
    // 構建行程卡片HTML
    let stopsHTML = '';
    itinerary.stops.forEach((stop, idx) => {
        const isLast = idx === itinerary.stops.length - 1;
        
        // 安全獲取數據
        const name = stop.name || 'Unknown';
        const rating = (stop.rating != null && !isNaN(stop.rating)) ? stop.rating.toFixed(1) : 'N/A';
        const category = stop.category || 'N/A';
        const duration = stop.duration || 60;
        const reviews = stop.reviews || 0;
        const reason = stop.reason || '';
        const order = stop.order || (idx + 1);
        
        stopsHTML += `
            <div class="stop-item">
                <div class="stop-order">${order}</div>
                <div class="stop-content">
                    <div class="stop-name">${name}</div>
                    <div class="stop-details">
                        <span><i class="fas fa-star"></i> ${rating}</span>
                        <span><i class="fas fa-tag"></i> ${category}</span>
                        <span><i class="fas fa-clock"></i> ${duration}分鐘</span>
                        ${reviews > 0 ? `<span><i class="fas fa-comment"></i> ${reviews.toLocaleString()}</span>` : ''}
                    </div>
                    ${reason ? `<div class="stop-reason">${reason}</div>` : ''}
                </div>
            </div>
            ${!isLast ? '<div style="text-align: center; color: #9ca3af; margin: 8px 0;">↓</div>' : ''}
        `;
    });
    
    // 構建提示HTML
    let tipsHTML = '';
    if (itinerary.tips && itinerary.tips.length > 0) {
        tipsHTML = `
            <div class="itinerary-tips">
                <h4><i class="fas fa-lightbulb"></i> 旅遊建議</h4>
                <ul>
                    ${itinerary.tips.map(tip => `<li>${tip}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    card.innerHTML = `
        <div class="itinerary-header">
            <div class="itinerary-icon">️</div>
            <div>
                <h3 class="itinerary-title">${itinerary.title || '旅遊行程'}</h3>
            </div>
        </div>
        
        <div class="itinerary-meta">
            <div class="meta-item">
                <i class="fas fa-map-marker-alt"></i>
                <span>${itinerary.total_stops} 個景點</span>
            </div>
            <div class="meta-item">
                <i class="fas fa-clock"></i>
                <span>${itinerary.total_duration} 分鐘</span>
            </div>
            <div class="meta-item">
                <i class="fas fa-route"></i>
                <span>${itinerary.total_distance.toFixed(1)} km</span>
            </div>
            ${itinerary.route.optimized ? '<div class="meta-item"><i class="fas fa-check-circle"></i><span style="color: #10b981;">路徑已優化</span></div>' : ''}
        </div>
        
        <div class="itinerary-stops">
            ${stopsHTML}
        </div>
        
        ${itinerary.summary ? `
            <div class="itinerary-summary">
                <h4><i class="fas fa-file-alt"></i> 行程摘要</h4>
                <p>${itinerary.summary}</p>
            </div>
        ` : ''}
        
        ${tipsHTML}
    `;
    
    container.appendChild(card);
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
    // 創建 Leaflet 結果地圖
    if (!map) {
        map = new LeafletResultMap('map');
    }
    
    // 保存到全局變量供卡片點擊使用
    window.map = map;
    
    // 設置數據並繪製
    map.setData(data);
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
    
    // 添加點擊卡片顯示路徑功能
    card.addEventListener('click', function(e) {
        // 如果點擊的是座標區域，則執行複製而不顯示路徑
        if (e.target.closest('.coord-value')) {
            return;
        }
        
        // 顯示該 POI 的路徑
        if (window.map && typeof window.map.showSinglePOIRoute === 'function') {
            window.map.showSinglePOIRoute(rank - 1);
            
            // 高亮當前卡片
            document.querySelectorAll('.recommendation-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        }
    });
    
    card.style.cursor = 'pointer';
    
    // 添加點擊複製座標功能
    const coordValue = card.querySelector('.coord-value');
    if (coordValue) {
        coordValue.style.cursor = 'pointer';
        coordValue.addEventListener('click', function(e) {
            e.stopPropagation(); // 阻止事件冒泡到卡片
            const coordText = `${poi.latitude}, ${poi.longitude}`;
            navigator.clipboard.writeText(coordText).then(() => {
                // 顯示複製成功提示
                const originalTitle = coordValue.getAttribute('title');
                coordValue.setAttribute('title', '已複製! ');
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
