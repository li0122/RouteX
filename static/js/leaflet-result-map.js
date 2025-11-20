/**
 * Leaflet 結果地圖
 * 顯示推薦結果和 OSRM 路線
 */
class LeafletResultMap {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.map = null;
        this.markers = [];
        this.routePolyline = null;
        this.legendControl = null;
        
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
        
        // 初始化 Leaflet 地圖
        this.map = L.map(this.containerId, {
            center: this.options.center,
            zoom: this.options.zoom,
            minZoom: this.options.minZoom,
            maxZoom: this.options.maxZoom
        });
        
        // 添加瓦片層（支援離線 MBTiles 或在線 OSM）
        if (this.options.useMBTiles && typeof L.tileLayer.mbTiles !== 'undefined') {
            console.log('️ 使用離線 MBTiles 瓦片');
            L.tileLayer.mbTiles(this.options.mbtilesPath, {
                attribution: '© OpenStreetMap contributors (離線)',
                minZoom: this.options.minZoom,
                maxZoom: this.options.maxZoom
            }).addTo(this.map);
        } else {
            console.log(' 使用在線 OSM 瓦片');
            L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 19
            }).addTo(this.map);
        }
    }

    setData(data, mode = 'poi') {
        // 清除舊數據
        this.clearMap();
        
        const { start_location, end_location, recommendations } = data;
        
        this.startLocation = start_location;
        this.endLocation = end_location;
        this.pois = recommendations;
        this.mode = mode; // 保存模式：'poi' 或 'itinerary'
        
        // 添加標記
        this.addStartMarker(start_location);
        this.addEndMarker(end_location);
        this.addPOIMarkers(recommendations);
        
        // 繪製簡單路線
        this.drawSimpleRoute(start_location, recommendations, end_location);
        
        // 調整視圖
        this.fitBounds();
        
        // 異步加載 OSRM 路線
        this.fetchAndDrawOSRM(start_location, recommendations, end_location);
    }
    
    addStartMarker(location) {
        const marker = L.marker(location, {
            icon: L.divIcon({
                html: `<div class="custom-marker" style="background: #10b981; width: 32px; height: 32px; 
                            border-radius: 50%; display: flex; align-items: center; justify-content: center;
                            border: 4px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-size: 20px;">
                            
                       </div>`,
                className: '',
                iconSize: [32, 32],
                iconAnchor: [16, 16]
            }),
            zIndexOffset: 1000
        }).addTo(this.map);
        
        marker.bindPopup(`
            <div class="poi-popup">
                <h3> 出發點</h3>
                <p><strong>座標:</strong> ${location[0].toFixed(6)}, ${location[1].toFixed(6)}</p>
            </div>
        `);
        
        this.markers.push(marker);
    }
    
    addEndMarker(location) {
        const marker = L.marker(location, {
            icon: L.divIcon({
                html: `<div class="custom-marker" style="background: #ef4444; width: 32px; height: 32px; 
                            border-radius: 50%; display: flex; align-items: center; justify-content: center;
                            border: 4px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-size: 20px;">
                            
                       </div>`,
                className: '',
                iconSize: [32, 32],
                iconAnchor: [16, 16]
            }),
            zIndexOffset: 1000
        }).addTo(this.map);
        
        marker.bindPopup(`
            <div class="poi-popup">
                <h3> 目的地</h3>
                <p><strong>座標:</strong> ${location[0].toFixed(6)}, ${location[1].toFixed(6)}</p>
            </div>
        `);
        
        this.markers.push(marker);
    }
    
    addPOIMarkers(recommendations) {
        recommendations.forEach((rec, index) => {
            const poi = rec.poi;
            const marker = L.marker([poi.latitude, poi.longitude], {
                icon: L.divIcon({
                    html: `<div class="poi-marker" style="background: #3b82f6; color: white; width: 36px; height: 36px; 
                                border-radius: 50%; display: flex; align-items: center; justify-content: center;
                                font-weight: bold; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                                font-size: 16px;">
                                ${index + 1}
                           </div>`,
                    className: '',
                    iconSize: [36, 36],
                    iconAnchor: [18, 18]
                }),
                zIndexOffset: 500
            }).addTo(this.map);
            
            // 彈出視窗內容
            const popupContent = `
                <div class="poi-popup">
                    <h3>${poi.name}</h3>
                    <div style="margin: 10px 0;">
                        <p><strong>️ 類別:</strong> ${poi.primary_category || 'N/A'}</p>
                        <p><strong>⭐ 評分:</strong> ${poi.avg_rating ? poi.avg_rating.toFixed(1) : 'N/A'} 
                           (${poi.num_reviews || 0} 評論)</p>
                        <p><strong> 推薦分數:</strong> ${rec.score.toFixed(3)}</p>
                        <p><strong>⏱️ 額外時間:</strong> ${rec.extra_time_minutes ? rec.extra_time_minutes.toFixed(1) : 'N/A'} 分鐘</p>
                        <p><strong> 座標:</strong> ${poi.latitude.toFixed(6)}, ${poi.longitude.toFixed(6)}</p>
                        ${rec.llm_approved ? '<p><strong> AI 審核通過</strong></p>' : ''}
                    </div>
                </div>
            `;
            
            marker.bindPopup(popupContent, {
                maxWidth: 300
            });
            
            this.markers.push(marker);
        });
    }
    
    drawSimpleRoute(start, recommendations, end) {
        // 單點推薦模式：不自動繪製路徑，等待用戶點擊
        if (this.mode === 'poi') {
            console.log(' 單點推薦模式：路徑隱藏，點擊卡片顯示');
            return;
        }
        
        // 行程推薦模式：繪製簡單路線
        const points = [
            start,
            ...recommendations.map(r => [r.poi.latitude, r.poi.longitude]),
            end
        ];
        
        this.routeLine = L.polyline(points, {
            color: '#9ca3af',
            weight: 3,
            opacity: 0.5,
            dashArray: '10, 10'
        }).addTo(this.map);
        
        console.log(' 繪製簡單路線（行程模式）');
    }
    
    async showSinglePOIRoute(poiIndex) {
        // 清除之前的路線
        if (this.currentRoute) {
            this.map.removeLayer(this.currentRoute);
            this.currentRoute = null;
        }
        
        if (this.currentRouteInfo) {
            this.map.removeLayer(this.currentRouteInfo);
            this.currentRouteInfo = null;
        }
        
        // 高亮選中的 POI marker
        this.highlightPOIMarker(poiIndex);
        
        const poi = this.pois[poiIndex];
        const poiLocation = [poi.poi.latitude, poi.poi.longitude];
        
        // 請求 OSRM 路線：起點 → POI → 終點
        try {
            console.log(` 請求 POI ${poiIndex + 1} 的路線...`);
            
            const waypoints = [this.startLocation, poiLocation, this.endLocation];
            
            const response = await fetch('/api/route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    waypoints: waypoints,
                    options: { geometries: 'geojson', overview: 'full' }
                }),
                signal: AbortSignal.timeout(10000)
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            if (data.code === 'Ok' && data.route && data.route.geometry) {
                const route = data.route;
                const coordinates = route.geometry.coordinates.map(c => [c[1], c[0]]);
                
                // 繪製路線
                this.currentRoute = L.polyline(coordinates, {
                    color: '#3b82f6',
                    weight: 5,
                    opacity: 0.9,
                    lineJoin: 'round',
                    lineCap: 'round'
                }).addTo(this.map);
                
                // 添加路線資訊
                const distance = (route.distance / 1000).toFixed(1);
                const duration = Math.round(route.duration / 60);
                
                const midPoint = coordinates[Math.floor(coordinates.length / 2)];
                this.currentRouteInfo = L.popup({
                    closeButton: false,
                    autoClose: false,
                    closeOnClick: false,
                    className: 'route-info-popup'
                })
                .setLatLng(midPoint)
                .setContent(`
                    <div style="text-align: center; padding: 8px;">
                        <strong style="font-size: 15px; color: white;">${poi.poi.name}</strong><br>
                        <span style="color: white; font-size: 14px; font-weight: 600;"> ${distance} km | ⏱️ ${duration} 分鐘</span>
                    </div>
                `)
                .addTo(this.map);
                
                // 調整視圖以顯示完整路線
                this.map.fitBounds(this.currentRoute.getBounds(), { padding: [50, 50] });
                
                console.log(` POI ${poiIndex + 1} 路線繪製完成`);
            }
        } catch (error) {
            console.error('路線請求失敗:', error);
        }
    }
    
    highlightPOIMarker(poiIndex) {
        // 重置所有 POI marker 樣式
        this.pois.forEach((_, idx) => {
            const marker = this.markers[idx + 2]; // 前兩個是起點和終點
            if (marker && marker._icon) {
                const markerDiv = marker._icon.querySelector('.poi-marker');
                if (markerDiv) {
                    markerDiv.style.background = '#3b82f6';
                    markerDiv.style.transform = 'scale(1)';
                }
            }
        });
        
        // 高亮選中的 marker
        const selectedMarker = this.markers[poiIndex + 2];
        if (selectedMarker && selectedMarker._icon) {
            const markerDiv = selectedMarker._icon.querySelector('.poi-marker');
            if (markerDiv) {
                markerDiv.style.background = '#f59e0b';
                markerDiv.style.transform = 'scale(1.2)';
            }
        }
    }
    
    async fetchAndDrawOSRM(start, recommendations, end) {
        // 單點推薦模式：不自動繪裭完整路線
        if (this.mode === 'poi') {
            console.log('⏸️ 單點推薦模式：跳過自動路線繪裭');
            return;
        }
        
        // 行程推薦模式：繪裭 OSRM 路線
        try {
            console.log(' 開始請求 OSRM 路線...');
            
            const waypoints = [
                start,
                ...recommendations.map(r => [r.poi.latitude, r.poi.longitude]),
                end
            ];
            
            // 調用本地 API 代理
            const response = await fetch('/api/route', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    waypoints: waypoints,
                    options: {
                        geometries: 'geojson',
                        overview: 'full'
                    }
                }),
                signal: AbortSignal.timeout(20000)
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.code === 'Ok' && data.route && data.route.geometry) {
                const route = data.route;
                const coordinates = route.geometry.coordinates.map(c => [c[1], c[0]]);
                
                // 移除簡單路線
                if (this.routeLine) {
                    this.map.removeLayer(this.routeLine);
                    this.routeLine = null;
                }
                
                // 繪製 OSRM 路線
                this.osrmLine = L.polyline(coordinates, {
                    color: '#3b82f6',
                    weight: 4,
                    opacity: 0.8,
                    lineJoin: 'round',
                    lineCap: 'round'
                }).addTo(this.map);
                
                // 添加路線資訊彈出視窗
                const distance = (route.distance / 1000).toFixed(1);
                const duration = Math.round(route.duration / 60);
                
                const midPoint = coordinates[Math.floor(coordinates.length / 2)];
                L.popup()
                    .setLatLng(midPoint)
                    .setContent(`
                        <div style="text-align: center;">
                            <strong>️ 路線資訊</strong><br>
                            <p> 距離: <strong>${distance} km</strong></p>
                            <p>⏱️ 時間: <strong>${duration} 分鐘</strong></p>
                            <p> 路線點: ${coordinates.length}</p>
                        </div>
                    `)
                    .openOn(this.map);
                
                console.log(` OSRM 路線加載成功！`);
                console.log(`   距離: ${distance} km`);
                console.log(`   時間: ${duration} 分鐘`);
                console.log(`   路線點數: ${coordinates.length}`);
                
            } else {
                throw new Error(`路線錯誤: ${data.error || 'Unknown'}`);
            }
            
        } catch (error) {
            console.error(' OSRM 路線加載失敗:', error.message);
            console.warn('️ 使用簡單路線顯示');
        }
    }
    
    clearMap() {
        // 移除所有標記
        this.markers.forEach(marker => this.map.removeLayer(marker));
        this.markers = [];
        
        // 移除路線
        if (this.routeLine) {
            this.map.removeLayer(this.routeLine);
            this.routeLine = null;
        }
        
        if (this.osrmLine) {
            this.map.removeLayer(this.osrmLine);
            this.osrmLine = null;
        }
    }
    
    fitBounds() {
        if (this.markers.length > 0) {
            const bounds = L.latLngBounds(
                this.markers.map(m => m.getLatLng())
            );
            this.map.fitBounds(bounds, {
                padding: [50, 50],
                maxZoom: 14
            });
        }
    }
    
    addLegend() {
        const Legend = L.Control.extend({
            options: {
                position: 'bottomright'
            },
            
            onAdd: function() {
                const div = L.DomUtil.create('div', 'map-legend');
                div.innerHTML = `
                    <div style="background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 8px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 13px;">
                        <h4 style="margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">圖例</h4>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 16px; height: 16px; background: #10b981; border-radius: 50%; 
                                        margin-right: 8px; border: 2px solid white;"></div>
                            <span>出發點</span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 16px; height: 16px; background: #ef4444; border-radius: 50%; 
                                        margin-right: 8px; border: 2px solid white;"></div>
                            <span>目的地</span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 16px; height: 16px; background: #3b82f6; border-radius: 50%; 
                                        margin-right: 8px; border: 2px solid white;"></div>
                            <span>推薦景點</span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 30px; height: 3px; background: #3b82f6; margin-right: 8px;"></div>
                            <span>實際路線</span>
                        </div>
                    </div>
                `;
                return div;
            }
        });
        
        this.map.addControl(new Legend());
    }
}
