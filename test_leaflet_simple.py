"""
簡單的 Flask 測試伺服器 - 僅用於測試 Leaflet 地圖
不載入完整推薦系統
"""

from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')


@app.route('/test_leaflet')
def test_leaflet():
    """Leaflet 地圖測試頁面"""
    return render_template('test_leaflet.html')


if __name__ == '__main__':
    print("=" * 60)
    print("🗺️  Leaflet 測試伺服器")
    print("=" * 60)
    print("訪問以下地址測試地圖：")
    print("  📍 http://localhost:5050/test_leaflet")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5050,
        debug=True
    )
