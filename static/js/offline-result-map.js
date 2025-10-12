// 離線結果地圖顯示器
// 使用 Canvas 繪製推薦路線和 POI

class OfflineResultMap {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.bounds = options.bounds || {
            minLat: 37.6,
            maxLat: 37.9,
            minLng: -122.6,
            maxLng: -122.2
        };
        
        this.startLocation = null;
        this.endLocation = null;
        this.pois = [];
        
        this.init();
    }
    
    init() {
        // 清空容器
        this.container.innerHTML = '';
        
        // 設置 Canvas 大小
        const containerWidth = this.container.offsetWidth;
        this.canvas.width = containerWidth;
        this.canvas.height = 500;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '500px';
        this.canvas.style.borderRadius = '10px';
        this.canvas.style.border = '2px solid #e5e7eb';
        
        this.container.appendChild(this.canvas);
    }
    
    setData(startLocation, endLocation, recommendations) {
        this.startLocation = startLocation;
        this.endLocation = endLocation;
        this.pois = recommendations.map(rec => ({
            lat: rec.poi.latitude,
            lng: rec.poi.longitude,
            name: rec.poi.name,
            category: rec.poi.primary_category,
            score: rec.score
        }));
        
        // 自動調整邊界
        this.autoAdjustBounds();
        
        this.draw();
    }
    
    autoAdjustBounds() {
        const allPoints = [this.startLocation, this.endLocation, ...this.pois.map(p => [p.lat, p.lng])];
        const lats = allPoints.map(p => p[0]);
        const lngs = allPoints.map(p => p[1]);
        
        const minLat = Math.min(...lats);
        const maxLat = Math.max(...lats);
        const minLng = Math.min(...lngs);
        const maxLng = Math.max(...lngs);
        
        // 添加邊距
        const latPadding = (maxLat - minLat) * 0.2;
        const lngPadding = (maxLng - minLng) * 0.2;
        
        this.bounds = {
            minLat: minLat - latPadding,
            maxLat: maxLat + latPadding,
            minLng: minLng - lngPadding,
            maxLng: maxLng + lngPadding
        };
    }
    
    draw() {
        // 清除畫布
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 繪製背景
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, '#e0f2fe');
        gradient.addColorStop(1, '#dbeafe');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 繪製網格
        this.drawGrid();
        
        // 繪製路線
        this.drawRoute();
        
        // 繪製標記
        this.drawMarkers();
        
        // 繪製圖例
        this.drawLegend();
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#cbd5e1';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([2, 2]);
        
        // 垂直線
        for (let i = 0; i <= 10; i++) {
            const x = (this.canvas.width / 10) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // 水平線
        for (let i = 0; i <= 10; i++) {
            const y = (this.canvas.height / 10) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        this.ctx.setLineDash([]);
    }
    
    drawRoute() {
        if (!this.startLocation || !this.endLocation) return;
        
        const points = [
            this.startLocation,
            ...this.pois.map(p => [p.lat, p.lng]),
            this.endLocation
        ];
        
        // 繪製路線
        this.ctx.beginPath();
        const firstPoint = this.latLngToPixel(points[0][0], points[0][1]);
        this.ctx.moveTo(firstPoint.x, firstPoint.y);
        
        for (let i = 1; i < points.length; i++) {
            const point = this.latLngToPixel(points[i][0], points[i][1]);
            this.ctx.lineTo(point.x, point.y);
        }
        
        this.ctx.strokeStyle = '#2563eb';
        this.ctx.lineWidth = 4;
        this.ctx.setLineDash([10, 5]);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }
    
    drawMarkers() {
        // 繪製起點
        if (this.startLocation) {
            const pos = this.latLngToPixel(this.startLocation[0], this.startLocation[1]);
            this.drawMarker(pos, '#10b981', '起點', '▶');
        }
        
        // 繪製終點
        if (this.endLocation) {
            const pos = this.latLngToPixel(this.endLocation[0], this.endLocation[1]);
            this.drawMarker(pos, '#ef4444', '終點', '🏁');
        }
        
        // 繪製 POI
        this.pois.forEach((poi, index) => {
            const pos = this.latLngToPixel(poi.lat, poi.lng);
            this.drawPOIMarker(pos, poi, index + 1);
        });
    }
    
    drawMarker(position, color, label, icon) {
        // 繪製標記圓圈
        this.ctx.beginPath();
        this.ctx.arc(position.x, position.y, 20, 0, 2 * Math.PI);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 4;
        this.ctx.stroke();
        
        // 繪製圖標
        this.ctx.fillStyle = 'white';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(icon, position.x, position.y);
        
        // 繪製標籤背景
        this.ctx.font = 'bold 11px Arial';
        const metrics = this.ctx.measureText(label);
        const padding = 6;
        const labelWidth = metrics.width + padding * 2;
        const labelHeight = 18;
        const labelX = position.x - labelWidth / 2;
        const labelY = position.y - 35;
        
        this.ctx.fillStyle = color;
        this.ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
        
        // 繪製標籤文字
        this.ctx.fillStyle = 'white';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(label, position.x, labelY + labelHeight / 2 + 1);
    }
    
    drawPOIMarker(position, poi, number) {
        // 繪製標記圓圈
        this.ctx.beginPath();
        this.ctx.arc(position.x, position.y, 18, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#2563eb';
        this.ctx.fill();
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        
        // 繪製編號
        this.ctx.fillStyle = 'white';
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(number, position.x, position.y);
    }
    
    drawLegend() {
        const legendX = 20;
        const legendY = 20;
        const legendWidth = 180;
        const legendHeight = 100;
        
        // 繪製背景
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        this.ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
        this.ctx.strokeStyle = '#cbd5e1';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);
        
        // 繪製標題
        this.ctx.fillStyle = '#1f2937';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText('圖例', legendX + 10, legendY + 20);
        
        // 繪製圖例項目
        const items = [
            { color: '#10b981', text: '起點', y: 40 },
            { color: '#ef4444', text: '終點', y: 60 },
            { color: '#2563eb', text: '推薦地點', y: 80 }
        ];
        
        items.forEach(item => {
            // 繪製圓圈
            this.ctx.beginPath();
            this.ctx.arc(legendX + 20, legendY + item.y, 8, 0, 2 * Math.PI);
            this.ctx.fillStyle = item.color;
            this.ctx.fill();
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // 繪製文字
            this.ctx.fillStyle = '#4b5563';
            this.ctx.font = '11px Arial';
            this.ctx.fillText(item.text, legendX + 35, legendY + item.y + 3);
        });
    }
    
    latLngToPixel(lat, lng) {
        const x = ((lng - this.bounds.minLng) / (this.bounds.maxLng - this.bounds.minLng)) * this.canvas.width;
        const y = ((this.bounds.maxLat - lat) / (this.bounds.maxLat - this.bounds.minLat)) * this.canvas.height;
        return { x, y };
    }
}
