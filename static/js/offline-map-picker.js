/**// 離線地圖座標選擇器

 * 離線地圖選擇器 - 改進版// 使用 Canvas 繪製簡單的座標網格系統

 * 功能：縮放、平移、預設地標、懸停提示、距離計算

 */class OfflineMapPicker {

class OfflineMapPicker {    constructor(containerId, options = {}) {

    constructor(canvasId, options = {}) {        this.container = document.getElementById(containerId);

        this.canvas = document.getElementById(canvasId);        this.canvas = document.createElement('canvas');

        this.ctx = this.canvas.getContext('2d');        this.ctx = this.canvas.getContext('2d');

                

        // 地圖邊界 (舊金山灣區)        // 預設範圍：舊金山灣區

        this.bounds = {        this.bounds = options.bounds || {

            north: 37.9,            minLat: 37.6,

            south: 37.6,            maxLat: 37.9,

            west: -122.6,            minLng: -122.6,

            east: -122.2            maxLng: -122.2

        };        };

                

        // 標記狀態        // 標記

        this.startMarker = null;        this.startMarker = null;

        this.endMarker = null;        this.endMarker = null;

        this.currentMode = null; // 'start' or 'end'        this.isSelectingStart = false;

                this.isSelectingEnd = false;

        // 縮放和平移狀態        this.hoveredPoint = null;

        this.scale = 1.0;        

        this.minScale = 0.5;        // 縮放和平移

        this.maxScale = 3.0;        this.scale = 1;

        this.offsetX = 0;        this.offsetX = 0;

        this.offsetY = 0;        this.offsetY = 0;

                this.isDragging = false;

        // 拖動狀態        this.dragStartX = 0;

        this.isDragging = false;        this.dragStartY = 0;

        this.dragStartX = 0;        

        this.dragStartY = 0;        // 回調函數

        this.lastOffsetX = 0;        this.onStartSelect = options.onStartSelect || (() => {});

        this.lastOffsetY = 0;        this.onEndSelect = options.onEndSelect || (() => {});

                

        // 懸停狀態        // 預設地點

        this.mouseX = 0;        this.presetLocations = [

        this.mouseY = 0;            { name: '舊金山市中心', lat: 37.7749, lng: -122.4194, icon: '🏙️', color: '#f59e0b' },

        this.showHoverTip = false;            { name: '金門大橋', lat: 37.8199, lng: -122.4783, icon: '🌉', color: '#ef4444' },

                    { name: '漁人碼頭', lat: 37.8080, lng: -122.4177, icon: '⛵', color: '#3b82f6' },

        // 預設地標            { name: '金門公園', lat: 37.7694, lng: -122.4862, icon: '🌳', color: '#10b981' },

        this.presetLocations = [            { name: '雙子峰', lat: 37.7544, lng: -122.4477, icon: '⛰️', color: '#8b5cf6' },

            { name: '舊金山市中心', lat: 37.7749, lng: -122.4194, icon: '🏙️', color: '#ff6b35' },            { name: '聯合廣場', lat: 37.7880, lng: -122.4075, icon: '🏛️', color: '#ec4899' },

            { name: '金門大橋', lat: 37.8199, lng: -122.4783, icon: '🌉', color: '#e63946' },        ];

            { name: '漁人碼頭', lat: 37.8080, lng: -122.4177, icon: '⛵', color: '#457b9d' },        

            { name: '金門公園', lat: 37.7694, lng: -122.4862, icon: '🌳', color: '#2a9d8f' },        this.init();

            { name: '雙子峰', lat: 37.7544, lng: -122.4477, icon: '⛰️', color: '#9d4edd' },    }

            { name: '聯合廣場', lat: 37.7880, lng: -122.4075, icon: '🏛️', color: '#f72585' }    

        ];    init() {

                // 設置 Canvas 大小

        // 回調函數        const containerWidth = this.container.offsetWidth;

        this.onLocationSelected = options.onLocationSelected || (() => {});        this.canvas.width = containerWidth;

                this.canvas.height = 500;

        this.init();        this.canvas.style.width = '100%';

    }        this.canvas.style.height = '500px';

            this.canvas.style.cursor = 'crosshair';

    init() {        this.canvas.style.borderRadius = '10px';

        // 設置 canvas 大小        this.canvas.style.border = '2px solid #e5e7eb';

        this.canvas.width = this.canvas.offsetWidth;        

        this.canvas.height = 500; // 增加高度以提供更好的視野        this.container.appendChild(this.canvas);

                

        // 綁定事件        // 添加事件監聽器

        this.canvas.addEventListener('click', this.handleClick.bind(this));        this.canvas.addEventListener('click', (e) => this.handleClick(e));

        this.canvas.addEventListener('wheel', this.handleWheel.bind(this));        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));

        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));

        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));

        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseLeave(e));

        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));

                

        // 初始繪製        // 初始繪製

        this.draw();        this.draw();

    }    }

        

    // 座標轉換：經緯度 -> 像素    draw() {

    latLngToPixel(lat, lng) {        // 清除畫布

        const x = ((lng - this.bounds.west) / (this.bounds.east - this.bounds.west)) * this.canvas.width;        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const y = ((this.bounds.north - lat) / (this.bounds.north - this.bounds.south)) * this.canvas.height;        

                this.ctx.save();

        // 應用縮放和平移        this.ctx.translate(this.offsetX, this.offsetY);

        const scaledX = x * this.scale + this.offsetX;        this.ctx.scale(this.scale, this.scale);

        const scaledY = y * this.scale + this.offsetY;        

                // 繪製背景

        return { x: scaledX, y: scaledY };        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);

    }        gradient.addColorStop(0, '#e0f2fe');

            gradient.addColorStop(0.5, '#f0f9ff');

    // 座標轉換：像素 -> 經緯度        gradient.addColorStop(1, '#dbeafe');

    pixelToLatLng(x, y) {        this.ctx.fillStyle = gradient;

        // 補償縮放和平移        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const unscaledX = (x - this.offsetX) / this.scale;        

        const unscaledY = (y - this.offsetY) / this.scale;        // 繪製網格

                this.drawGrid();

        const lng = this.bounds.west + (unscaledX / this.canvas.width) * (this.bounds.east - this.bounds.west);        

        const lat = this.bounds.north - (unscaledY / this.canvas.height) * (this.bounds.north - this.bounds.south);        // 繪製預設地點

                this.drawPresetLocations();

        return { lat, lng };        

    }        // 繪製標記

            if (this.startMarker) {

    // 計算兩點距離 (公里)            this.drawMarker(this.startMarker, '#10b981', '起點', '📍');

    calculateDistance(lat1, lng1, lat2, lng2) {        }

        const R = 6371; // 地球半徑 (km)        if (this.endMarker) {

        const dLat = (lat2 - lat1) * Math.PI / 180;            this.drawMarker(this.endMarker, '#ef4444', '終點', '🎯');

        const dLng = (lng2 - lng1) * Math.PI / 180;        }

        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +        

                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *        // 如果兩個標記都存在，繪製連線和距離

                  Math.sin(dLng/2) * Math.sin(dLng/2);        if (this.startMarker && this.endMarker) {

        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));            this.drawLine();

        return R * c;        }

    }        

            // 繪製懸停提示

    // 處理滾輪縮放        if (this.hoveredPoint) {

    handleWheel(e) {            this.drawHoverTooltip(this.hoveredPoint);

        e.preventDefault();        }

                

        const delta = e.deltaY > 0 ? -0.1 : 0.1;        this.ctx.restore();

        const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale + delta));        

                // 繪製控制說明（不受縮放影響）

        if (newScale === this.scale) return;        this.drawControls();

            }

        // 以鼠標位置為中心縮放    

        const rect = this.canvas.getBoundingClientRect();    drawGrid() {

        const mouseX = e.clientX - rect.left;        this.ctx.strokeStyle = 'rgba(203, 213, 225, 0.5)';

        const mouseY = e.clientY - rect.top;        this.ctx.lineWidth = 1 / this.scale;

                

        // 計算縮放前的世界座標        // 垂直線

        const worldX = (mouseX - this.offsetX) / this.scale;        const lngStep = (this.bounds.maxLng - this.bounds.minLng) / 10;

        const worldY = (mouseY - this.offsetY) / this.scale;        for (let i = 0; i <= 10; i++) {

                    const lng = this.bounds.minLng + lngStep * i;

        // 更新縮放            const pos = this.latLngToPixel(this.bounds.minLat, lng);

        this.scale = newScale;            this.ctx.beginPath();

                    this.ctx.moveTo(pos.x, 0);

        // 調整偏移以保持鼠標位置不變            this.ctx.lineTo(pos.x, this.canvas.height);

        this.offsetX = mouseX - worldX * this.scale;            this.ctx.stroke();

        this.offsetY = mouseY - worldY * this.scale;        }

                

        this.draw();        // 水平線

    }        const latStep = (this.bounds.maxLat - this.bounds.minLat) / 10;

            for (let i = 0; i <= 10; i++) {

    // 處理鼠標按下            const lat = this.bounds.minLat + latStep * i;

    handleMouseDown(e) {            const pos = this.latLngToPixel(lat, this.bounds.minLng);

        if (this.currentMode) return; // 選擇模式時不拖動            this.ctx.beginPath();

                    this.ctx.moveTo(0, pos.y);

        this.isDragging = true;            this.ctx.lineTo(this.canvas.width, pos.y);

        const rect = this.canvas.getBoundingClientRect();            this.ctx.stroke();

        this.dragStartX = e.clientX - rect.left;        }

        this.dragStartY = e.clientY - rect.top;        

        this.lastOffsetX = this.offsetX;        // 繪製座標標籤

        this.lastOffsetY = this.offsetY;        this.ctx.fillStyle = '#64748b';

        this.canvas.style.cursor = 'grabbing';        this.ctx.font = `${10 / this.scale}px Arial`;

    }        

            // 經度標籤（底部）

    // 處理鼠標移動        for (let i = 0; i <= 4; i++) {

    handleMouseMove(e) {            const lng = this.bounds.minLng + (this.bounds.maxLng - this.bounds.minLng) * (i / 4);

        const rect = this.canvas.getBoundingClientRect();            const pos = this.latLngToPixel(this.bounds.minLat, lng);

        this.mouseX = e.clientX - rect.left;            this.ctx.fillText(lng.toFixed(2), pos.x + 5, this.canvas.height - 5);

        this.mouseY = e.clientY - rect.top;        }

                

        if (this.isDragging) {        // 緯度標籤（左側）

            // 拖動地圖        for (let i = 0; i <= 4; i++) {

            const dx = this.mouseX - this.dragStartX;            const lat = this.bounds.maxLat - (this.bounds.maxLat - this.bounds.minLat) * (i / 4);

            const dy = this.mouseY - this.dragStartY;            const pos = this.latLngToPixel(lat, this.bounds.minLng);

            this.offsetX = this.lastOffsetX + dx;            this.ctx.fillText(lat.toFixed(2), 5, pos.y + 15);

            this.offsetY = this.lastOffsetY + dy;        }

            this.draw();    }

        } else if (this.currentMode) {    

            // 選擇模式：顯示懸停提示    drawPresetLocations() {

            this.showHoverTip = true;        this.presetLocations.forEach(location => {

            this.draw();            const pos = this.latLngToPixel(location.lat, location.lng);

        }            

    }            // 繪製背景圓圈

                this.ctx.beginPath();

    // 處理鼠標釋放            this.ctx.arc(pos.x, pos.y, 20 / this.scale, 0, 2 * Math.PI);

    handleMouseUp(e) {            this.ctx.fillStyle = location.color;

        this.isDragging = false;            this.ctx.globalAlpha = 0.2;

        this.canvas.style.cursor = this.currentMode ? 'crosshair' : 'default';            this.ctx.fill();

    }            this.ctx.globalAlpha = 1;

                

    // 處理鼠標離開            // 繪製圖標

    handleMouseLeave(e) {            this.ctx.font = `${16 / this.scale}px Arial`;

        this.isDragging = false;            this.ctx.textAlign = 'center';

        this.showHoverTip = false;            this.ctx.textBaseline = 'middle';

        this.canvas.style.cursor = 'default';            this.ctx.fillText(location.icon, pos.x, pos.y);

        this.draw();            

    }            // 繪製名稱

                this.ctx.font = `bold ${11 / this.scale}px Arial`;

    // 處理點擊            this.ctx.fillStyle = '#1f2937';

    handleClick(e) {            this.ctx.fillText(location.name, pos.x, pos.y + 25 / this.scale);

        if (this.isDragging) return;        });

        if (!this.currentMode) return;    }

            

        const rect = this.canvas.getBoundingClientRect();    drawMarker(position, color, label, icon) {

        const x = e.clientX - rect.left;        // 繪製陰影

        const y = e.clientY - rect.top;        this.ctx.beginPath();

                this.ctx.arc(position.x, position.y + 3 / this.scale, 18 / this.scale, 0, 2 * Math.PI);

        // 檢查是否點擊了預設地標        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';

        for (const location of this.presetLocations) {        this.ctx.fill();

            const pos = this.latLngToPixel(location.lat, location.lng);        

            const distance = Math.sqrt(Math.pow(x - pos.x, 2) + Math.pow(y - pos.y, 2));        // 繪製標記圓圈

                    this.ctx.beginPath();

            if (distance < 20) {        this.ctx.arc(position.x, position.y, 18 / this.scale, 0, 2 * Math.PI);

                // 點擊了地標        this.ctx.fillStyle = color;

                this.selectLocation(location.lat, location.lng, location.name);        this.ctx.fill();

                return;        this.ctx.strokeStyle = 'white';

            }        this.ctx.lineWidth = 4 / this.scale;

        }        this.ctx.stroke();

                

        // 點擊了空白處：轉換為經緯度        // 繪製圖標

        const coords = this.pixelToLatLng(x, y);        this.ctx.font = `${16 / this.scale}px Arial`;

                this.ctx.textAlign = 'center';

        // 檢查邊界        this.ctx.textBaseline = 'middle';

        if (coords.lat < this.bounds.south || coords.lat > this.bounds.north ||        this.ctx.fillStyle = 'white';

            coords.lng < this.bounds.west || coords.lng > this.bounds.east) {        this.ctx.fillText(icon, position.x, position.y);

            return;        

        }        // 繪製標籤背景

                this.ctx.font = `bold ${12 / this.scale}px Arial`;

        this.selectLocation(coords.lat, coords.lng);        const metrics = this.ctx.measureText(label);

    }        const padding = 8 / this.scale;

            const labelWidth = metrics.width + padding * 2;

    // 選擇位置        const labelHeight = 22 / this.scale;

    selectLocation(lat, lng, name = null) {        const labelX = position.x - labelWidth / 2;

        const marker = {        const labelY = position.y - 40 / this.scale;

            lat: lat,        

            lng: lng,        // 繪製標籤陰影

            name: name        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';

        };        this.ctx.fillRect(labelX + 2 / this.scale, labelY + 2 / this.scale, labelWidth, labelHeight);

                

        if (this.currentMode === 'start') {        // 繪製標籤背景

            this.startMarker = marker;        this.ctx.fillStyle = color;

            this.onLocationSelected('start', lat, lng);        this.ctx.fillRect(labelX, labelY, labelWidth, labelHeight);

        } else if (this.currentMode === 'end') {        

            this.endMarker = marker;        // 繪製標籤文字

            this.onLocationSelected('end', lat, lng);        this.ctx.fillStyle = 'white';

        }        this.ctx.textAlign = 'center';

                this.ctx.fillText(label, position.x, labelY + labelHeight / 2 + 1 / this.scale);

        this.currentMode = null;        

        this.showHoverTip = false;        // 繪製座標

        this.canvas.style.cursor = 'default';        const coordText = `${position.lat.toFixed(4)}, ${position.lng.toFixed(4)}`;

        this.draw();        this.ctx.font = `${10 / this.scale}px Arial`;

    }        this.ctx.fillStyle = '#6b7280';

            this.ctx.fillText(coordText, position.x, position.y + 30 / this.scale);

    // 繪製地圖    }

    draw() {    

        const ctx = this.ctx;    drawLine() {

        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);        // 繪製連線

                this.ctx.beginPath();

        // 背景漸層        this.ctx.moveTo(this.startMarker.x, this.startMarker.y);

        const gradient = ctx.createLinearGradient(0, 0, 0, this.canvas.height);        this.ctx.lineTo(this.endMarker.x, this.endMarker.y);

        gradient.addColorStop(0, '#e3f2fd');        this.ctx.strokeStyle = '#2563eb';

        gradient.addColorStop(0.5, '#bbdefb');        this.ctx.lineWidth = 3 / this.scale;

        gradient.addColorStop(1, '#90caf9');        this.ctx.setLineDash([10 / this.scale, 5 / this.scale]);

        ctx.fillStyle = gradient;        this.ctx.stroke();

        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);        this.ctx.setLineDash([]);

                

        // 繪製網格        // 計算並顯示距離

        this.drawGrid();        const distance = this.calculateDistance(

                    this.startMarker.lat, this.startMarker.lng,

        // 繪製預設地標            this.endMarker.lat, this.endMarker.lng

        this.drawPresetLocations();        );

                

        // 繪製路線（如果兩點都已選擇）        const midX = (this.startMarker.x + this.endMarker.x) / 2;

        if (this.startMarker && this.endMarker) {        const midY = (this.startMarker.y + this.endMarker.y) / 2;

            this.drawRoute();        

        }        this.ctx.font = `bold ${12 / this.scale}px Arial`;

                this.ctx.fillStyle = '#2563eb';

        // 繪製標記        this.ctx.textAlign = 'center';

        if (this.startMarker) {        this.ctx.fillText(`${distance.toFixed(2)} km`, midX, midY - 10 / this.scale);

            this.drawMarker(this.startMarker, 'start');    }

        }    

        if (this.endMarker) {    drawHoverTooltip(point) {

            this.drawMarker(this.endMarker, 'end');        const tooltipText = `點擊此處\n${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}`;

        }        

                this.ctx.font = `${11 / this.scale}px Arial`;

        // 繪製懸停提示        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';

        if (this.showHoverTip && this.currentMode) {        this.ctx.fillRect(

            this.drawHoverTip();            point.x + 10 / this.scale,

        }            point.y - 30 / this.scale,

                    100 / this.scale,

        // 繪製操作說明            40 / this.scale

        this.drawInstructions();        );

    }        

            this.ctx.fillStyle = 'white';

    // 繪製網格        this.ctx.textAlign = 'left';

    drawGrid() {        this.ctx.fillText('點擊此處', point.x + 15 / this.scale, point.y - 15 / this.scale);

        const ctx = this.ctx;        this.ctx.fillText(`${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}`, point.x + 15 / this.scale, point.y - 3 / this.scale);

        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';    }

        ctx.lineWidth = 1;    

            drawControls() {

        // 垂直線        const controls = [

        for (let i = 0; i <= 10; i++) {            '🖱️ 點擊選擇地點',

            const lng = this.bounds.west + (this.bounds.east - this.bounds.west) * i / 10;            '🔍 滾輪縮放',

            const pos = this.latLngToPixel(this.bounds.north, lng);            '✋ 按住拖動平移',

            ctx.beginPath();            '💡 點擊地標快速選擇'

            ctx.moveTo(pos.x, 0);        ];

            ctx.lineTo(pos.x, this.canvas.height);        

            ctx.stroke();        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';

                    this.ctx.fillRect(this.canvas.width - 180, 10, 170, 100);

            // 標籤        this.ctx.strokeStyle = '#cbd5e1';

            ctx.fillStyle = '#666';        this.ctx.lineWidth = 1;

            ctx.font = '10px Arial';        this.ctx.strokeRect(this.canvas.width - 180, 10, 170, 100);

            ctx.textAlign = 'center';        

            ctx.fillText(lng.toFixed(2), pos.x, this.canvas.height - 5);        this.ctx.fillStyle = '#1f2937';

        }        this.ctx.font = 'bold 11px Arial';

                this.ctx.textAlign = 'left';

        // 水平線        this.ctx.fillText('操作說明', this.canvas.width - 170, 28);

        for (let i = 0; i <= 10; i++) {        

            const lat = this.bounds.south + (this.bounds.north - this.bounds.south) * i / 10;        this.ctx.font = '10px Arial';

            const pos = this.latLngToPixel(lat, this.bounds.west);        this.ctx.fillStyle = '#4b5563';

            ctx.beginPath();        controls.forEach((text, i) => {

            ctx.moveTo(0, pos.y);            this.ctx.fillText(text, this.canvas.width - 170, 48 + i * 16);

            ctx.lineTo(this.canvas.width, pos.y);        });

            ctx.stroke();    }

                

            // 標籤    handleClick(event) {

            ctx.fillStyle = '#666';        if (this.isDragging) return;

            ctx.font = '10px Arial';        

            ctx.textAlign = 'left';        const rect = this.canvas.getBoundingClientRect();

            ctx.fillText(lat.toFixed(2), 5, pos.y - 5);        const x = (event.clientX - rect.left - this.offsetX) / this.scale;

        }        const y = (event.clientY - rect.top - this.offsetY) / this.scale;

    }        

            // 檢查是否點擊預設地點

    // 繪製預設地標        const clickedPreset = this.presetLocations.find(location => {

    drawPresetLocations() {            const pos = this.latLngToPixel(location.lat, location.lng);

        const ctx = this.ctx;            const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);

                    return distance < 20;

        for (const location of this.presetLocations) {        });

            const pos = this.latLngToPixel(location.lat, location.lng);        

                    let latLng;

            // 陰影        if (clickedPreset) {

            ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';            latLng = { lat: clickedPreset.lat, lng: clickedPreset.lng };

            ctx.shadowBlur = 5;        } else {

            ctx.shadowOffsetX = 2;            latLng = this.pixelToLatLng(x, y);

            ctx.shadowOffsetY = 2;        }

                    

            // 繪製圓形背景        if (this.isSelectingStart) {

            ctx.fillStyle = location.color;            const pos = this.latLngToPixel(latLng.lat, latLng.lng);

            ctx.beginPath();            this.startMarker = { x: pos.x, y: pos.y, lat: latLng.lat, lng: latLng.lng };

            ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);            this.isSelectingStart = false;

            ctx.fill();            this.onStartSelect(latLng.lat, latLng.lng);

                        this.draw();

            // 清除陰影        } else if (this.isSelectingEnd) {

            ctx.shadowColor = 'transparent';            const pos = this.latLngToPixel(latLng.lat, latLng.lng);

            ctx.shadowBlur = 0;            this.endMarker = { x: pos.x, y: pos.y, lat: latLng.lat, lng: latLng.lng };

            ctx.shadowOffsetX = 0;            this.isSelectingEnd = false;

            ctx.shadowOffsetY = 0;            this.onEndSelect(latLng.lat, latLng.lng);

                        this.draw();

            // 繪製圖標        }

            ctx.font = '16px Arial';    }

            ctx.textAlign = 'center';    

            ctx.textBaseline = 'middle';    handleMouseMove(event) {

            ctx.fillText(location.icon, pos.x, pos.y);        const rect = this.canvas.getBoundingClientRect();

                    const x = (event.clientX - rect.left - this.offsetX) / this.scale;

            // 繪製名稱（小字）        const y = (event.clientY - rect.top - this.offsetY) / this.scale;

            ctx.font = '9px Arial';        

            ctx.fillStyle = '#333';        if (this.isDragging) {

            ctx.textBaseline = 'top';            const dx = event.clientX - this.dragStartX;

            ctx.fillText(location.name, pos.x, pos.y + 15);            const dy = event.clientY - this.dragStartY;

        }            this.offsetX += dx;

    }            this.offsetY += dy;

                this.dragStartX = event.clientX;

    // 繪製標記            this.dragStartY = event.clientY;

    drawMarker(marker, type) {            this.draw();

        const ctx = this.ctx;            return;

        const pos = this.latLngToPixel(marker.lat, marker.lng);        }

                

        const color = type === 'start' ? '#4caf50' : '#f44336';        if (this.isSelectingStart || this.isSelectingEnd) {

        const label = type === 'start' ? '起點' : '終點';            const latLng = this.pixelToLatLng(x, y);

        const icon = type === 'start' ? '📍' : '🎯';            this.hoveredPoint = { x, y, lat: latLng.lat, lng: latLng.lng };

                    this.draw();

        // 陰影        }

        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';    }

        ctx.shadowBlur = 10;    

        ctx.shadowOffsetX = 3;    handleMouseDown(event) {

        ctx.shadowOffsetY = 3;        if (!this.isSelectingStart && !this.isSelectingEnd) {

                    this.isDragging = true;

        // 繪製圓形標記            this.dragStartX = event.clientX;

        ctx.fillStyle = color;            this.dragStartY = event.clientY;

        ctx.beginPath();            this.canvas.style.cursor = 'grabbing';

        ctx.arc(pos.x, pos.y, 18, 0, Math.PI * 2);        }

        ctx.fill();    }

            

        // 外圈    handleMouseUp(event) {

        ctx.strokeStyle = '#fff';        this.isDragging = false;

        ctx.lineWidth = 3;        this.canvas.style.cursor = 'crosshair';

        ctx.stroke();    }

            

        // 清除陰影    handleMouseLeave(event) {

        ctx.shadowColor = 'transparent';        this.isDragging = false;

        ctx.shadowBlur = 0;        this.hoveredPoint = null;

        ctx.shadowOffsetX = 0;        this.canvas.style.cursor = 'crosshair';

        ctx.shadowOffsetY = 0;        this.draw();

            }

        // 繪製圖標    

        ctx.font = '20px Arial';    handleWheel(event) {

        ctx.textAlign = 'center';        event.preventDefault();

        ctx.textBaseline = 'middle';        

        ctx.fillText(icon, pos.x, pos.y);        const rect = this.canvas.getBoundingClientRect();

                const mouseX = event.clientX - rect.left;

        // 標籤背景        const mouseY = event.clientY - rect.top;

        ctx.fillStyle = color;        

        ctx.fillRect(pos.x - 30, pos.y - 40, 60, 20);        const wheel = event.deltaY < 0 ? 1.1 : 0.9;

                const newScale = this.scale * wheel;

        // 標籤文字        

        ctx.fillStyle = '#fff';        if (newScale < 0.5 || newScale > 3) return;

        ctx.font = 'bold 12px Arial';        

        ctx.textBaseline = 'middle';        this.offsetX = mouseX - (mouseX - this.offsetX) * wheel;

        ctx.fillText(label, pos.x, pos.y - 30);        this.offsetY = mouseY - (mouseY - this.offsetY) * wheel;

                this.scale = newScale;

        // 顯示座標（小字）        

        ctx.fillStyle = '#666';        this.draw();

        ctx.font = '10px Arial';    }

        ctx.textBaseline = 'top';    

        const coordText = `${marker.lat.toFixed(4)}, ${marker.lng.toFixed(4)}`;    setSelectingStart(isSelecting) {

        ctx.fillText(coordText, pos.x, pos.y + 22);        this.isSelectingStart = isSelecting;

                this.isSelectingEnd = false;

        // 顯示名稱（如果有）        this.canvas.style.cursor = isSelecting ? 'crosshair' : 'default';

        if (marker.name) {        this.hoveredPoint = null;

            ctx.fillStyle = '#333';        this.draw();

            ctx.font = 'bold 11px Arial';    }

            ctx.fillText(marker.name, pos.x, pos.y + 34);    

        }    setSelectingEnd(isSelecting) {

    }        this.isSelectingEnd = isSelecting;

            this.isSelectingStart = false;

    // 繪製路線        this.canvas.style.cursor = isSelecting ? 'crosshair' : 'default';

    drawRoute() {        this.hoveredPoint = null;

        const ctx = this.ctx;        this.draw();

        const startPos = this.latLngToPixel(this.startMarker.lat, this.startMarker.lng);    }

        const endPos = this.latLngToPixel(this.endMarker.lat, this.endMarker.lng);    

            pixelToLatLng(x, y) {

        // 繪製虛線        const lat = this.bounds.maxLat - (y / this.canvas.height) * (this.bounds.maxLat - this.bounds.minLat);

        ctx.setLineDash([5, 5]);        const lng = this.bounds.minLng + (x / this.canvas.width) * (this.bounds.maxLng - this.bounds.minLng);

        ctx.strokeStyle = '#2196f3';        return { lat, lng };

        ctx.lineWidth = 2;    }

        ctx.beginPath();    

        ctx.moveTo(startPos.x, startPos.y);    latLngToPixel(lat, lng) {

        ctx.lineTo(endPos.x, endPos.y);        const x = ((lng - this.bounds.minLng) / (this.bounds.maxLng - this.bounds.minLng)) * this.canvas.width;

        ctx.stroke();        const y = ((this.bounds.maxLat - lat) / (this.bounds.maxLat - this.bounds.minLat)) * this.canvas.height;

        ctx.setLineDash([]);        return { x, y };

            }

        // 計算並顯示距離    

        const distance = this.calculateDistance(    calculateDistance(lat1, lng1, lat2, lng2) {

            this.startMarker.lat, this.startMarker.lng,        const R = 6371; // 地球半徑（公里）

            this.endMarker.lat, this.endMarker.lng        const dLat = (lat2 - lat1) * Math.PI / 180;

        );        const dLng = (lng2 - lng1) * Math.PI / 180;

                const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +

        const midX = (startPos.x + endPos.x) / 2;                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *

        const midY = (startPos.y + endPos.y) / 2;                  Math.sin(dLng / 2) * Math.sin(dLng / 2);

                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        // 距離標籤背景        return R * c;

        ctx.fillStyle = 'rgba(33, 150, 243, 0.9)';    }

        ctx.fillRect(midX - 40, midY - 12, 80, 24);    

            getStartLocation() {

        // 距離文字        return this.startMarker ? [this.startMarker.lat, this.startMarker.lng] : null;

        ctx.fillStyle = '#fff';    }

        ctx.font = 'bold 12px Arial';    

        ctx.textAlign = 'center';    getEndLocation() {

        ctx.textBaseline = 'middle';        return this.endMarker ? [this.endMarker.lat, this.endMarker.lng] : null;

        ctx.fillText(`${distance.toFixed(2)} km`, midX, midY);    }

    }}

        

    // 繪製懸停提示    draw() {

    drawHoverTip() {        // 清除畫布

        const ctx = this.ctx;        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const coords = this.pixelToLatLng(this.mouseX, this.mouseY);        

                // 繪製背景

        // 檢查邊界        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);

        if (coords.lat < this.bounds.south || coords.lat > this.bounds.north ||        gradient.addColorStop(0, '#e0f2fe');

            coords.lng < this.bounds.west || coords.lng > this.bounds.east) {        gradient.addColorStop(1, '#dbeafe');

            return;        this.ctx.fillStyle = gradient;

        }        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

                

        const text = `${coords.lat.toFixed(4)}, ${coords.lng.toFixed(4)}`;        // 繪製網格

        const padding = 8;        this.drawGrid();

        const textWidth = ctx.measureText(text).width;        

                // 繪製地標文字

        // 背景        this.drawLandmarks();

        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';        

        ctx.fillRect(this.mouseX + 10, this.mouseY - 25, textWidth + padding * 2, 20);        // 繪製標記

                if (this.startMarker) {

        // 文字            this.drawMarker(this.startMarker, '#10b981', '起點');

        ctx.fillStyle = '#fff';        }

        ctx.font = '11px Arial';        if (this.endMarker) {

        ctx.textAlign = 'left';            this.drawMarker(this.endMarker, '#ef4444', '終點');

        ctx.textBaseline = 'middle';        }

        ctx.fillText(text, this.mouseX + 10 + padding, this.mouseY - 15);        

                // 如果兩個標記都存在，繪製連線

        // 提示文字        if (this.startMarker && this.endMarker) {

        ctx.font = '10px Arial';            this.drawLine();

        ctx.fillStyle = '#ffeb3b';        }

        ctx.fillText('點擊選擇此位置', this.mouseX + 10 + padding, this.mouseY - 5);    }

    }    

        drawGrid() {

    // 繪製操作說明        this.ctx.strokeStyle = '#cbd5e1';

    drawInstructions() {        this.ctx.lineWidth = 1;

        const ctx = this.ctx;        

        const x = this.canvas.width - 10;        // 垂直線

        const y = 10;        for (let i = 0; i <= 10; i++) {

                    const x = (this.canvas.width / 10) * i;

        // 背景            this.ctx.beginPath();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';            this.ctx.moveTo(x, 0);

        ctx.fillRect(x - 160, y, 150, 85);            this.ctx.lineTo(x, this.canvas.height);

                    this.ctx.stroke();

        // 邊框        }

        ctx.strokeStyle = '#2196f3';        

        ctx.lineWidth = 2;        // 水平線

        ctx.strokeRect(x - 160, y, 150, 85);        for (let i = 0; i <= 10; i++) {

                    const y = (this.canvas.height / 10) * i;

        // 文字            this.ctx.beginPath();

        ctx.fillStyle = '#333';            this.ctx.moveTo(0, y);

        ctx.font = '11px Arial';            this.ctx.lineTo(this.canvas.width, y);

        ctx.textAlign = 'left';            this.ctx.stroke();

        ctx.textBaseline = 'top';        }

                

        const instructions = [        // 繪製座標標籤

            '🖱️ 點擊選擇地點',        this.ctx.fillStyle = '#64748b';

            '🔍 滾輪縮放地圖',        this.ctx.font = '10px Arial';

            '✋ 按住拖動平移',        

            '💡 點擊地標快速選擇'        // 經度標籤

        ];        for (let i = 0; i <= 4; i++) {

                    const lng = this.bounds.minLng + (this.bounds.maxLng - this.bounds.minLng) * (i / 4);

        instructions.forEach((text, i) => {            const x = (this.canvas.width / 4) * i;

            ctx.fillText(text, x - 150, y + 10 + i * 18);            this.ctx.fillText(lng.toFixed(2), x + 5, this.canvas.height - 5);

        });        }

    }        

            // 緯度標籤

    // 公共方法：開始選擇起點        for (let i = 0; i <= 4; i++) {

    selectStart() {            const lat = this.bounds.maxLat - (this.bounds.maxLat - this.bounds.minLat) * (i / 4);

        this.currentMode = 'start';            const y = (this.canvas.height / 4) * i;

        this.canvas.style.cursor = 'crosshair';            this.ctx.fillText(lat.toFixed(2), 5, y + 15);

    }        }

        }

    // 公共方法：開始選擇終點    

    selectEnd() {    drawLandmarks() {

        this.currentMode = 'end';        // 添加一些地標標註

        this.canvas.style.cursor = 'crosshair';        const landmarks = [

    }            { name: '舊金山市中心', lat: 37.7749, lng: -122.4194, icon: '🏙️' },

                { name: '金門大橋', lat: 37.8199, lng: -122.4783, icon: '🌉' },

    // 公共方法：清除標記            { name: '漁人碼頭', lat: 37.8080, lng: -122.4177, icon: '⛵' },

    clearMarkers() {            { name: '金門公園', lat: 37.7694, lng: -122.4862, icon: '🌳' },

        this.startMarker = null;        ];

        this.endMarker = null;        

        this.currentMode = null;        this.ctx.font = 'bold 12px Arial';

        this.showHoverTip = false;        landmarks.forEach(landmark => {

        this.canvas.style.cursor = 'default';            const pos = this.latLngToPixel(landmark.lat, landmark.lng);

        this.draw();            

    }            // 繪製圖標

                this.ctx.fillStyle = '#1f2937';

    // 公共方法：重置視圖            this.ctx.fillText(landmark.icon, pos.x - 8, pos.y - 8);

    resetView() {            

        this.scale = 1.0;            // 繪製名稱

        this.offsetX = 0;            this.ctx.font = '10px Arial';

        this.offsetY = 0;            this.ctx.fillStyle = '#4b5563';

        this.draw();            this.ctx.fillText(landmark.name, pos.x + 8, pos.y + 5);

    }            this.ctx.font = 'bold 12px Arial';

}        });

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
