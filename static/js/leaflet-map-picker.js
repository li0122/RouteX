/**
 * Leaflet 地圖選擇器
 * 用於選擇起點和終點
 */
class LeafletMapPicker {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.map = null;
        this.startMarker = null;
        this.endMarker = null;
        this.startLocation = null;
        this.endLocation = null;
        this.landmarks = [];
        this.routeLine = null;
        this.modeIndicator = null;
        
        // 預設選項
        this.options = {
            center: [37.7749, -122.4194],
            zoom: 12,
            minZoom: 8,
            maxZoom: 18,
            useMBTiles: true,  // 是否使用離線 MBTiles
            mbtilesPath: '/static/data/sf_bay_area.mbtiles',
            ...options
        };
        
        this.init();
    }
    
    init() {
        // 創建地圖容器
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`容器 ${this.containerId} 不存在`);
            return;
        }
        
        // 設置容器樣式
        container.style.height = '500px';
        container.style.border = '2px solid #e5e7eb';
        container.style.borderRadius = '8px';
        container.style.overflow = 'hidden';
        
        // 初始化 Leaflet 地圖
        this.map = L.map(this.containerId, {
            center: this.options.center,
            zoom: this.options.zoom,
            minZoom: this.options.minZoom,
            maxZoom: this.options.maxZoom
        });
        
        // 添加瓦片層（支援離線 MBTiles 或在線 OSM）
        if (this.options.useMBTiles && typeof L.tileLayer.mbTiles !== 'undefined') {
            console.log('🗺️ 使用離線 MBTiles 瓦片');
            L.tileLayer.mbTiles(this.options.mbtilesPath, {
                attribution: '© OpenStreetMap contributors (離線)',
                minZoom: this.options.minZoom,
                maxZoom: this.options.maxZoom
            }).addTo(this.map);
        } else {
            console.log('🌐 使用在線 OSM 瓦片');
            L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 19
            }).addTo(this.map);
        }
        
        // 添加知名地標
        this.addLandmarks();
        
        // 監聽地圖點擊
        this.map.on('click', (e) => this.handleMapClick(e));
        
        console.log('✅ Leaflet 地圖選擇器初始化完成');
    }
    
    addLandmarks() {
        const landmarks = [
            { name: '金門大橋', lat: 37.8199, lng: -122.4783, icon: '🌉' },
            { name: '漁人碼頭', lat: 37.8080, lng: -122.4177, icon: '🎣' },
            { name: '聯合廣場', lat: 37.7880, lng: -122.4075, icon: '🏛️' },
            { name: '惡魔島', lat: 37.8267, lng: -122.4230, icon: '🏝️' },
            { name: '金門公園', lat: 37.7694, lng: -122.4862, icon: '🌳' },
            { name: '九曲花街', lat: 37.8021, lng: -122.4187, icon: '🌸' }
        ];
        
        landmarks.forEach(landmark => {
            const marker = L.marker([landmark.lat, landmark.lng], {
                icon: L.divIcon({
                    html: `<div style="font-size: 24px; text-shadow: 0 0 3px white;">${landmark.icon}</div>`,
                    className: 'landmark-marker',
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                })
            }).addTo(this.map);
            
            marker.bindPopup(`
                <div style="text-align: center;">
                    <strong style="font-size: 16px;">${landmark.name}</strong><br>
                    <small style="color: #6b7280;">點擊地圖設定起終點</small><br>
                    <button onclick="mapPicker.quickSetStart(${landmark.lat}, ${landmark.lng})" 
                            style="margin: 5px; padding: 5px 10px; background: #10b981; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        設為起點
                    </button>
                    <button onclick="mapPicker.quickSetEnd(${landmark.lat}, ${landmark.lng})"
                            style="margin: 5px; padding: 5px 10px; background: #ef4444; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        設為終點
                    </button>
                </div>
            `);
        });
    }
    
    setMode(mode) {
        this.mode = mode;
        this.map.getContainer().style.cursor = 'crosshair';
        
        // 顯示提示訊息
        const modeText = mode === 'start' ? '出發點' : '目的地';
        const color = mode === 'start' ? '#10b981' : '#ef4444';
        
        // 創建臨時提示框
        if (this.modeIndicator) {
            this.map.removeControl(this.modeIndicator);
        }
        
        const ModeIndicator = L.Control.extend({
            onAdd: function() {
                const div = L.DomUtil.create('div', 'mode-indicator');
                div.innerHTML = `
                    <div style="background: ${color}; color: white; padding: 10px 20px; 
                                border-radius: 8px; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                        點擊地圖選擇${modeText}
                    </div>
                `;
                return div;
            }
        });
        
        this.modeIndicator = new ModeIndicator({ position: 'topright' });
        this.modeIndicator.addTo(this.map);
    }
    
    handleMapClick(e) {
        if (!this.mode) return;
        
        const { lat, lng } = e.latlng;
        
        if (this.mode === 'start') {
            this.setStartLocation(lat, lng);
        } else if (this.mode === 'end') {
            this.setEndLocation(lat, lng);
        }
        
        // 清除模式
        this.mode = null;
        this.map.getContainer().style.cursor = '';
        if (this.modeIndicator) {
            this.map.removeControl(this.modeIndicator);
            this.modeIndicator = null;
        }
    }
    
    quickSetStart(lat, lng) {
        this.setStartLocation(lat, lng);
        this.map.closePopup();
    }
    
    quickSetEnd(lat, lng) {
        this.setEndLocation(lat, lng);
        this.map.closePopup();
    }
    
    setStartLocation(lat, lng) {
        // 移除舊標記
        if (this.startMarker) {
            this.map.removeLayer(this.startMarker);
        }
        
        this.startLocation = [lat, lng];
        
        // 創建起點標記
        this.startMarker = L.marker([lat, lng], {
            icon: L.divIcon({
                html: `<div style="background: #10b981; width: 24px; height: 24px; border-radius: 50%; 
                              border: 4px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>`,
                className: 'start-marker',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }),
            zIndexOffset: 1000
        }).addTo(this.map);
        
        this.startMarker.bindPopup(`
            <div style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">🟢</div>
                <strong>出發點</strong><br>
                <small>${lat.toFixed(6)}, ${lng.toFixed(6)}</small>
            </div>
        `).openPopup();
        
        // 更新顯示
        document.getElementById('startLocationDisplay').textContent = 
            `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        document.getElementById('startLocation').value = `${lat},${lng}`;
        
        // 更新路線
        this.updateRouteLine();
        
        console.log('✅ 出發點已設定:', lat, lng);
    }
    
    setEndLocation(lat, lng) {
        // 移除舊標記
        if (this.endMarker) {
            this.map.removeLayer(this.endMarker);
        }
        
        this.endLocation = [lat, lng];
        
        // 創建終點標記
        this.endMarker = L.marker([lat, lng], {
            icon: L.divIcon({
                html: `<div style="background: #ef4444; width: 24px; height: 24px; border-radius: 50%; 
                              border: 4px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>`,
                className: 'end-marker',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }),
            zIndexOffset: 1000
        }).addTo(this.map);
        
        this.endMarker.bindPopup(`
            <div style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">🔴</div>
                <strong>目的地</strong><br>
                <small>${lat.toFixed(6)}, ${lng.toFixed(6)}</small>
            </div>
        `).openPopup();
        
        // 更新顯示
        document.getElementById('endLocationDisplay').textContent = 
            `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        document.getElementById('endLocation').value = `${lat},${lng}`;
        
        // 更新路線
        this.updateRouteLine();
        
        console.log('✅ 目的地已設定:', lat, lng);
    }
    
    updateRouteLine() {
        // 移除舊路線
        if (this.routeLine) {
            this.map.removeLayer(this.routeLine);
        }
        
        // 如果起終點都設定了，繪製路線
        if (this.startLocation && this.endLocation) {
            this.routeLine = L.polyline(
                [this.startLocation, this.endLocation],
                {
                    color: '#3b82f6',
                    weight: 3,
                    opacity: 0.6,
                    dashArray: '10, 10'
                }
            ).addTo(this.map);
            
            // 調整視圖包含起終點
            const bounds = L.latLngBounds([this.startLocation, this.endLocation]);
            this.map.fitBounds(bounds, {
                padding: [80, 80],
                maxZoom: 13
            });
            
            // 計算距離
            const distance = this.map.distance(this.startLocation, this.endLocation) / 1000;
            console.log(`📏 直線距離: ${distance.toFixed(2)} km`);
        }
    }
}
