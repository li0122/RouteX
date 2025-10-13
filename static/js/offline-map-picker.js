/**
 * 離線地圖選擇器 - Canvas版本
 * 功能：縮放、平移、預設地標、懸停提示、距離計算
 */
class OfflineMapPicker {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Container not found:', containerId);
            return;
        }

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
        this.hoveredPoint = null;

        // 縮放和平移
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;

        // 回調函數
        this.onStartSelect = options.onStartSelect || (() => {});
        this.onEndSelect = options.onEndSelect || (() => {});

        // 預設地點
        this.presetLocations = [
            { name: '舊金山市中心', lat: 37.7749, lng: -122.4194, icon: '🏙️', color: '#f59e0b' },
            { name: '金門大橋', lat: 37.8199, lng: -122.4783, icon: '🌉', color: '#ef4444' },
            { name: '漁人碼頭', lat: 37.8080, lng: -122.4177, icon: '⛵', color: '#3b82f6' },
            { name: '金門公園', lat: 37.7694, lng: -122.4862, icon: '🌳', color: '#10b981' },
            { name: '雙子峰', lat: 37.7544, lng: -122.4477, icon: '⛰️', color: '#8b5cf6' },
            { name: '聯合廣場', lat: 37.7880, lng: -122.4075, icon: '🏛️', color: '#ec4899' },
        ];

        this.init();
    }

    init() {
        // 創建 Canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 800;
        this.canvas.height = 500;
        this.canvas.style.width = '100%';
        this.canvas.style.height = 'auto';
        this.canvas.style.border = '2px solid #e5e7eb';
        this.canvas.style.borderRadius = '8px';
        this.canvas.style.cursor = 'crosshair';
        this.canvas.style.background = '#f9fafb';
        this.container.appendChild(this.canvas);

        this.ctx = this.canvas.getContext('2d');

        // 綁定事件
        this.canvas.addEventListener('click', this.handleClick.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this), { passive: false });

        // 初始繪製
        this.draw();
    }

    latLngToCanvas(lat, lng) {
        const { minLat, maxLat, minLng, maxLng } = this.bounds;
        const x = (lng - minLng) / (maxLng - minLng);
        const y = 1 - (lat - minLat) / (maxLat - minLat);
        const canvasX = x * this.canvas.width * this.scale + this.offsetX;
        const canvasY = y * this.canvas.height * this.scale + this.offsetY;
        return { x: canvasX, y: canvasY };
    }

    canvasToLatLng(canvasX, canvasY) {
        const { minLat, maxLat, minLng, maxLng } = this.bounds;
        const x = (canvasX - this.offsetX) / (this.canvas.width * this.scale);
        const y = (canvasY - this.offsetY) / (this.canvas.height * this.scale);
        const lng = minLng + x * (maxLng - minLng);
        const lat = maxLat - y * (maxLat - minLat);
        return { lat, lng };
    }

    draw() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f0f9ff';
        ctx.fillRect(0, 0, width, height);

        this.drawGrid();
        this.drawPresetLocations();

        if (this.startMarker) {
            this.drawMarker(this.startMarker.lat, this.startMarker.lng, '🟢', '#10b981', '出發點');
        }
        if (this.endMarker) {
            this.drawMarker(this.endMarker.lat, this.endMarker.lng, '🔴', '#ef4444', '目的地');
        }

        if (this.startMarker && this.endMarker) {
            this.drawLine(this.startMarker, this.endMarker);
        }

        if (this.hoveredPoint) {
            this.drawHoverTooltip(this.hoveredPoint);
        }
    }

    drawGrid() {
        const ctx = this.ctx;
        const { minLat, maxLat, minLng, maxLng } = this.bounds;

        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;

        const lngStep = 0.1;
        for (let lng = minLng; lng <= maxLng; lng += lngStep) {
            const { x: x1, y: y1 } = this.latLngToCanvas(minLat, lng);
            const { x: x2, y: y2 } = this.latLngToCanvas(maxLat, lng);
            
            if (x1 >= 0 && x1 <= this.canvas.width) {
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();

                ctx.fillStyle = '#6b7280';
                ctx.font = '10px sans-serif';
                ctx.fillText(lng.toFixed(1), x1 - 15, this.canvas.height - 5);
            }
        }

        const latStep = 0.1;
        for (let lat = minLat; lat <= maxLat; lat += latStep) {
            const { x: x1, y: y1 } = this.latLngToCanvas(lat, minLng);
            const { x: x2, y: y2 } = this.latLngToCanvas(lat, maxLng);
            
            if (y1 >= 0 && y1 <= this.canvas.height) {
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();

                ctx.fillStyle = '#6b7280';
                ctx.font = '10px sans-serif';
                ctx.fillText(lat.toFixed(1), 5, y1 + 3);
            }
        }
    }

    drawPresetLocations() {
        const ctx = this.ctx;
        
        this.presetLocations.forEach(location => {
            const { x, y } = this.latLngToCanvas(location.lat, location.lng);
            
            if (x < 0 || x > this.canvas.width || y < 0 || y > this.canvas.height) {
                return;
            }

            ctx.fillStyle = location.color;
            ctx.beginPath();
            ctx.arc(x, y, 12, 0, Math.PI * 2);
            ctx.fill();

            ctx.font = '16px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(location.icon, x, y);

            ctx.font = 'bold 11px sans-serif';
            ctx.fillStyle = '#1f2937';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(location.name, x, y + 15);
        });
    }

    drawMarker(lat, lng, icon, color, label) {
        const ctx = this.ctx;
        const { x, y } = this.latLngToCanvas(lat, lng);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.font = 'bold 18px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(icon, x, y);

        ctx.font = 'bold 12px sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(label, x, y + 20);
    }

    drawLine(start, end) {
        const ctx = this.ctx;
        const p1 = this.latLngToCanvas(start.lat, start.lng);
        const p2 = this.latLngToCanvas(end.lat, end.lng);

        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.setLineDash([]);

        const distance = this.calculateDistance(start.lat, start.lng, end.lat, end.lng);
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;

        ctx.fillStyle = '#6366f1';
        ctx.fillRect(midX - 40, midY - 12, 80, 24);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(distance.toFixed(1) + ' km', midX, midY);
    }

    drawHoverTooltip(point) {
        const ctx = this.ctx;
        const text = point.lat.toFixed(4) + ', ' + point.lng.toFixed(4);
        
        ctx.font = '12px sans-serif';
        const metrics = ctx.measureText(text);
        const padding = 8;
        const width = metrics.width + padding * 2;
        const height = 24;

        let x = point.canvasX + 10;
        let y = point.canvasY - 30;

        if (x + width > this.canvas.width) x = point.canvasX - width - 10;
        if (y < 0) y = point.canvasY + 20;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x, y, width, height);

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x + width / 2, y + height / 2);
    }

    calculateDistance(lat1, lng1, lat2, lng2) {
        const R = 6371;
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLng = (lng2 - lng1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLng / 2) * Math.sin(dLng / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    handleClick(e) {
        if (this.isDragging) return;

        const rect = this.canvas.getBoundingClientRect();
        const canvasX = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const canvasY = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        const { lat, lng } = this.canvasToLatLng(canvasX, canvasY);

        for (const location of this.presetLocations) {
            const pos = this.latLngToCanvas(location.lat, location.lng);
            const dist = Math.sqrt(Math.pow(canvasX - pos.x, 2) + Math.pow(canvasY - pos.y, 2));
            
            if (dist < 15) {
                if (this.isSelectingStart) {
                    this.startMarker = { lat: location.lat, lng: location.lng };
                    this.onStartSelect(location.lat, location.lng);
                    this.isSelectingStart = false;
                } else if (this.isSelectingEnd) {
                    this.endMarker = { lat: location.lat, lng: location.lng };
                    this.onEndSelect(location.lat, location.lng);
                    this.isSelectingEnd = false;
                }
                this.draw();
                return;
            }
        }

        if (lat < this.bounds.minLat || lat > this.bounds.maxLat || 
            lng < this.bounds.minLng || lng > this.bounds.maxLng) {
            return;
        }

        if (this.isSelectingStart) {
            this.startMarker = { lat, lng };
            this.onStartSelect(lat, lng);
            this.isSelectingStart = false;
        } else if (this.isSelectingEnd) {
            this.endMarker = { lat, lng };
            this.onEndSelect(lat, lng);
            this.isSelectingEnd = false;
        }

        this.draw();
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const canvasY = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        if (this.isDragging) {
            const dx = e.clientX - this.dragStartX;
            const dy = e.clientY - this.dragStartY;
            this.offsetX = this.dragStartOffsetX + dx * (this.canvas.width / rect.width);
            this.offsetY = this.dragStartOffsetY + dy * (this.canvas.height / rect.height);
            this.draw();
            return;
        }

        const { lat, lng } = this.canvasToLatLng(canvasX, canvasY);
        if (lat >= this.bounds.minLat && lat <= this.bounds.maxLat &&
            lng >= this.bounds.minLng && lng <= this.bounds.maxLng) {
            this.hoveredPoint = { lat, lng, canvasX, canvasY };
        } else {
            this.hoveredPoint = null;
        }

        this.draw();
    }

    handleMouseDown(e) {
        this.isDragging = true;
        this.dragStartX = e.clientX;
        this.dragStartY = e.clientY;
        this.dragStartOffsetX = this.offsetX;
        this.dragStartOffsetY = this.offsetY;
        this.canvas.style.cursor = 'grabbing';
    }

    handleMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = this.isSelectingStart || this.isSelectingEnd ? 'crosshair' : 'grab';
    }

    handleMouseLeave(e) {
        this.isDragging = false;
        this.hoveredPoint = null;
        this.canvas.style.cursor = 'crosshair';
        this.draw();
    }

    handleWheel(e) {
        e.preventDefault();
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.5, Math.min(3, this.scale * delta));

        this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / this.scale);
        this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / this.scale);
        this.scale = newScale;

        this.draw();
    }

    setSelectingStart(value) {
        this.isSelectingStart = value;
        this.isSelectingEnd = false;
        this.canvas.style.cursor = value ? 'crosshair' : 'grab';
    }

    setSelectingEnd(value) {
        this.isSelectingEnd = value;
        this.isSelectingStart = false;
        this.canvas.style.cursor = value ? 'crosshair' : 'grab';
    }

    reset() {
        this.startMarker = null;
        this.endMarker = null;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.draw();
    }
}
