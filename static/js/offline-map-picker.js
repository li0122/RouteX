// 離線地圖座標選擇器
// 使用 Canvas 繪製簡單的座標網格系統

class OfflineMapPicker {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // 預設範圍：舊金山灣區
        this.bounds = options.bounds || {
            minLat: 37.6,
            maxLat: 37.9,
            minLng: -122.6,
            maxLng: -122.2
        };
        
        // 標記
        this.startMarker = null;
        this.endMarker = null;
        this.isSelectingStart = false;
        this.isSelectingEnd = false;
        
        // 回調函數
        this.onStartSelect = options.onStartSelect || (() => {});
        this.onEndSelect = options.onEndSelect || (() => {});
        
        this.init();
    }
    
    init() {
        // 設置 Canvas 大小
        const containerWidth = this.container.offsetWidth;
        this.canvas.width = containerWidth;
        this.canvas.height = 400;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '400px';
        this.canvas.style.cursor = 'crosshair';
        this.canvas.style.borderRadius = '10px';
        this.canvas.style.border = '2px solid #e5e7eb';
        
        this.container.appendChild(this.canvas);
        
        // 添加點擊事件
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        
        // 初始繪製
        this.draw();
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
        
        // 繪製地標文字
        this.drawLandmarks();
        
        // 繪製標記
        if (this.startMarker) {
            this.drawMarker(this.startMarker, '#10b981', '起點');
        }
        if (this.endMarker) {
            this.drawMarker(this.endMarker, '#ef4444', '終點');
        }
        
        // 如果兩個標記都存在，繪製連線
        if (this.startMarker && this.endMarker) {
            this.drawLine();
        }
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#cbd5e1';
        this.ctx.lineWidth = 1;
        
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
        
        // 繪製座標標籤
        this.ctx.fillStyle = '#64748b';
        this.ctx.font = '10px Arial';
        
        // 經度標籤
        for (let i = 0; i <= 4; i++) {
            const lng = this.bounds.minLng + (this.bounds.maxLng - this.bounds.minLng) * (i / 4);
            const x = (this.canvas.width / 4) * i;
            this.ctx.fillText(lng.toFixed(2), x + 5, this.canvas.height - 5);
        }
        
        // 緯度標籤
        for (let i = 0; i <= 4; i++) {
            const lat = this.bounds.maxLat - (this.bounds.maxLat - this.bounds.minLat) * (i / 4);
            const y = (this.canvas.height / 4) * i;
            this.ctx.fillText(lat.toFixed(2), 5, y + 15);
        }
    }
    
    drawLandmarks() {
        // 添加一些地標標註
        const landmarks = [
            { name: '舊金山市中心', lat: 37.7749, lng: -122.4194, icon: '🏙️' },
            { name: '金門大橋', lat: 37.8199, lng: -122.4783, icon: '🌉' },
            { name: '漁人碼頭', lat: 37.8080, lng: -122.4177, icon: '⛵' },
            { name: '金門公園', lat: 37.7694, lng: -122.4862, icon: '🌳' },
        ];
        
        this.ctx.font = 'bold 12px Arial';
        landmarks.forEach(landmark => {
            const pos = this.latLngToPixel(landmark.lat, landmark.lng);
            
            // 繪製圖標
            this.ctx.fillStyle = '#1f2937';
            this.ctx.fillText(landmark.icon, pos.x - 8, pos.y - 8);
            
            // 繪製名稱
            this.ctx.font = '10px Arial';
            this.ctx.fillStyle = '#4b5563';
            this.ctx.fillText(landmark.name, pos.x + 8, pos.y + 5);
            this.ctx.font = 'bold 12px Arial';
        });
    }
    
    drawMarker(position, color, label) {
        // 繪製標記圓圈
        this.ctx.beginPath();
        this.ctx.arc(position.x, position.y, 15, 0, 2 * Math.PI);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        
        // 繪製標籤
        this.ctx.fillStyle = color;
        this.ctx.font = 'bold 12px Arial';
        const metrics = this.ctx.measureText(label);
        this.ctx.fillRect(position.x - metrics.width / 2 - 5, position.y - 35, metrics.width + 10, 20);
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(label, position.x - metrics.width / 2, position.y - 20);
    }
    
    drawLine() {
        this.ctx.beginPath();
        this.ctx.moveTo(this.startMarker.x, this.startMarker.y);
        this.ctx.lineTo(this.endMarker.x, this.endMarker.y);
        this.ctx.strokeStyle = '#2563eb';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }
    
    handleClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // 轉換為經緯度
        const latLng = this.pixelToLatLng(x, y);
        
        if (this.isSelectingStart) {
            this.startMarker = { x, y, lat: latLng.lat, lng: latLng.lng };
            this.isSelectingStart = false;
            this.onStartSelect(latLng.lat, latLng.lng);
            this.draw();
        } else if (this.isSelectingEnd) {
            this.endMarker = { x, y, lat: latLng.lat, lng: latLng.lng };
            this.isSelectingEnd = false;
            this.onEndSelect(latLng.lat, latLng.lng);
            this.draw();
        }
    }
    
    setSelectingStart(isSelecting) {
        this.isSelectingStart = isSelecting;
        this.isSelectingEnd = false;
        this.canvas.style.cursor = isSelecting ? 'crosshair' : 'default';
    }
    
    setSelectingEnd(isSelecting) {
        this.isSelectingEnd = isSelecting;
        this.isSelectingStart = false;
        this.canvas.style.cursor = isSelecting ? 'crosshair' : 'default';
    }
    
    pixelToLatLng(x, y) {
        const lat = this.bounds.maxLat - (y / this.canvas.height) * (this.bounds.maxLat - this.bounds.minLat);
        const lng = this.bounds.minLng + (x / this.canvas.width) * (this.bounds.maxLng - this.bounds.minLng);
        return { lat, lng };
    }
    
    latLngToPixel(lat, lng) {
        const x = ((lng - this.bounds.minLng) / (this.bounds.maxLng - this.bounds.minLng)) * this.canvas.width;
        const y = ((this.bounds.maxLat - lat) / (this.bounds.maxLat - this.bounds.minLat)) * this.canvas.height;
        return { x, y };
    }
    
    getStartLocation() {
        return this.startMarker ? [this.startMarker.lat, this.startMarker.lng] : null;
    }
    
    getEndLocation() {
        return this.endMarker ? [this.endMarker.lat, this.endMarker.lng] : null;
    }
}
