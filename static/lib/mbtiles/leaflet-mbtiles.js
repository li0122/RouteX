/**
 * Leaflet MBTiles Plugin
 * 支援在瀏覽器中載入離線 MBTiles 瓦片包
 * 使用 sql.js 在瀏覽器端讀取 SQLite 資料庫
 */

(function() {
    // 確保 Leaflet 和 SQL.js 已載入
    if (typeof L === 'undefined') {
        throw new Error('Leaflet 必須先載入');
    }

    L.TileLayer.MBTiles = L.TileLayer.extend({
        options: {
            minZoom: 0,
            maxZoom: 18,
            tileSize: 256
        },

        initialize: function(mbtilesFile, options) {
            L.TileLayer.prototype.initialize.call(this, '', options);
            
            this._mbtilesFile = mbtilesFile;
            this._db = null;
            this._isReady = false;
            this._queue = [];
            
            this._loadMBTiles();
        },

        _loadMBTiles: function() {
            console.log('📦 載入 MBTiles 檔案:', this._mbtilesFile);
            
            fetch(this._mbtilesFile)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`無法載入 MBTiles: ${response.statusText}`);
                    }
                    return response.arrayBuffer();
                })
                .then(buffer => {
                    console.log('✅ MBTiles 檔案已載入，大小:', (buffer.byteLength / 1024 / 1024).toFixed(2), 'MB');
                    return this._initDatabase(buffer);
                })
                .then(() => {
                    console.log('✅ MBTiles 資料庫已初始化');
                    this._isReady = true;
                    this._processQueue();
                })
                .catch(error => {
                    console.error('❌ 載入 MBTiles 失敗:', error);
                    this.fire('tileerror', { error });
                });
        },

        _initDatabase: function(buffer) {
            return new Promise((resolve, reject) => {
                // 使用 SQL.js 初始化資料庫
                if (typeof initSqlJs === 'undefined') {
                    reject(new Error('SQL.js 未載入'));
                    return;
                }

                initSqlJs({
                    locateFile: file => `/static/lib/mbtiles/${file}`
                }).then(SQL => {
                    try {
                        const uint8Array = new Uint8Array(buffer);
                        this._db = new SQL.Database(uint8Array);
                        
                        // 讀取元數據
                        const metadataQuery = this._db.exec("SELECT name, value FROM metadata");
                        if (metadataQuery.length > 0) {
                            const metadata = {};
                            metadataQuery[0].values.forEach(([name, value]) => {
                                metadata[name] = value;
                            });
                            console.log('📋 MBTiles 元數據:', metadata);
                            
                            // 更新選項
                            if (metadata.minzoom) this.options.minZoom = parseInt(metadata.minzoom);
                            if (metadata.maxzoom) this.options.maxZoom = parseInt(metadata.maxzoom);
                        }
                        
                        resolve();
                    } catch (error) {
                        reject(error);
                    }
                }).catch(reject);
            });
        },

        getTileUrl: function(coords) {
            // 返回一個占位符 URL，實際瓦片通過 createTile 載入
            return `mbtiles://${coords.z}/${coords.x}/${coords.y}`;
        },

        createTile: function(coords, done) {
            const tile = document.createElement('img');
            
            if (this._isReady) {
                this._loadTile(coords, tile, done);
            } else {
                // 如果資料庫還未準備好，加入隊列
                this._queue.push({ coords, tile, done });
            }
            
            return tile;
        },

        _loadTile: function(coords, tile, done) {
            try {
                // MBTiles 使用 TMS 座標系統 (Y 軸翻轉)
                const z = coords.z;
                const x = coords.x;
                const y = (Math.pow(2, z) - 1) - coords.y;
                
                // 查詢瓦片
                const query = `SELECT tile_data FROM tiles WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?`;
                const result = this._db.exec(query, [z, x, y]);
                
                if (result.length > 0 && result[0].values.length > 0) {
                    const tileData = result[0].values[0][0];
                    
                    // 將二進制數據轉換為 Blob URL
                    const blob = new Blob([tileData], { type: 'image/png' });
                    const url = URL.createObjectURL(blob);
                    
                    tile.onload = () => {
                        URL.revokeObjectURL(url);
                        done(null, tile);
                    };
                    
                    tile.onerror = () => {
                        URL.revokeObjectURL(url);
                        done(new Error('瓦片載入失敗'), tile);
                    };
                    
                    tile.src = url;
                } else {
                    // 瓦片不存在，使用空白瓦片
                    done(new Error('瓦片不存在'), tile);
                }
            } catch (error) {
                console.error(`❌ 載入瓦片失敗 [${coords.z}/${coords.x}/${coords.y}]:`, error);
                done(error, tile);
            }
        },

        _processQueue: function() {
            // 處理隊列中的瓦片請求
            this._queue.forEach(({ coords, tile, done }) => {
                this._loadTile(coords, tile, done);
            });
            this._queue = [];
        },

        onRemove: function(map) {
            // 清理資料庫
            if (this._db) {
                this._db.close();
                this._db = null;
            }
            L.TileLayer.prototype.onRemove.call(this, map);
        }
    });

    // 工廠函數
    L.tileLayer.mbTiles = function(mbtilesFile, options) {
        return new L.TileLayer.MBTiles(mbtilesFile, options);
    };
})();
