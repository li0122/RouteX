"""
簡化的LLM過濾器
專門用於過濾旅遊推薦中不適合旅客的POI
"""

import requests
import json
import time
from typing import List, Dict, Optional, Any


class SimpleLLMFilter:
    """簡化的LLM過濾器 - 專注旅遊相關性"""
    
    def __init__(
        self,
        base_url: str = "140.125.248.15:31008",
        model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
        timeout: int = 30,
        delay_between_requests: float = 0.5
    ):
        self.base_url = f"http://{base_url}" if not base_url.startswith('http') else base_url
        self.model = model
        self.timeout = timeout
        self.delay_between_requests = delay_between_requests
        
        # 設置會話
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        print(f"🤖 LLM過濾器初始化完成")
        print(f"   端點: {self.base_url}")
        print(f"   模型: {self.model}")
    
    def is_travel_relevant(self, poi: Dict[str, Any], user_categories: Optional[List[str]] = None) -> bool:
        """
        判斷POI是否適合旅客
        
        Args:
            poi: POI資訊字典
            user_categories: 用戶偏好的類別列表（可選）
            
        Returns:
            True if 適合旅客, False otherwise
        """
        try:
            # 構建判斷提示
            prompt = self._build_travel_relevance_prompt(poi, user_categories)
            
            # 調用LLM
            response = self._call_llm(prompt)
            
            if response:
                # 解析回應
                return self._parse_travel_relevance_response(response)
            
            # 失敗時的預設判斷
            return self._fallback_travel_filter(poi)
            
        except Exception as e:
            print(f"   LLM判斷失敗: {e}")
            return self._fallback_travel_filter(poi)
    
    def filter_travel_pois(self, pois: List[Dict[str, Any]], user_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量過濾POI列表
        
        Args:
            pois: POI列表
            user_categories: 用戶偏好的類別列表（可選）
            
        Returns:
            過濾後的POI列表
        """
        if not pois:
            return []
        
        print(f"🔍 開始LLM旅遊相關性過濾...")
        print(f"   輸入POI數量: {len(pois)}")
        if user_categories:
            print(f"   用戶偏好類別: {', '.join(user_categories)}")
        
        filtered_pois = []
        rejected_count = 0
        
        for i, poi in enumerate(pois, 1):
            poi_name = poi.get('name', '未知POI')
            poi_category = poi.get('primary_category', '未分類')
            
            print(f"   ({i}/{len(pois)}) 檢查: {poi_name} ({poi_category})")
            
            if self.is_travel_relevant(poi, user_categories):
                filtered_pois.append(poi)
                print(f"     ✅ 通過 - 適合旅客")
            else:
                rejected_count += 1
                print(f"     ❌ 拒絕 - 不適合旅客")
            
            # 控制請求頻率
            if i < len(pois):  # 不是最後一個
                time.sleep(self.delay_between_requests)
        
        print(f"✅ LLM過濾完成!")
        print(f"   通過: {len(filtered_pois)} 個")
        print(f"   拒絕: {rejected_count} 個")
        
        return filtered_pois
    
    def sequential_llm_filter_top_k(
        self, 
        ranked_pois: List[Dict[str, Any]], 
        target_k: int,
        multiplier: int = 3,
        user_categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        按排序逐一審核，直到收集到target_k個通過的POI
        
        這是您要求的核心功能：
        1. 從第1名開始逐一審核
        2. 通過LLM審核的加入最終列表
        3. 直到收集到TOP K個為止
        
        Args:
            ranked_pois: 已排序的POI列表
            target_k: 目標數量
            multiplier: 初始搜索倍數（搜索前 target_k * multiplier 個）
            user_categories: 用戶偏好的類別列表（可選）
            
        Returns:
            通過LLM審核的TOP K POI列表
        """
        if not ranked_pois:
            return []
        
        print(f"🎯 開始逐一LLM審核流程")
        print(f"   目標: TOP {target_k} 推薦")
        print(f"   輸入: {len(ranked_pois)} 個排序POI")
        print(f"   審核範圍: 全部 {len(ranked_pois)} 個候選（不早停）")
        if user_categories:
            print(f"   用戶偏好類別: {', '.join(user_categories)}")
        
        approved_pois = []
        search_limit = min(len(ranked_pois), target_k * multiplier)
        
        print()
        
        # 從第1名開始逐一審核 - 不早停，審核完所有候選
        for rank, poi in enumerate(ranked_pois, 1):
            poi_name = poi.get('name', '未知POI')
            poi_category = poi.get('primary_category', '未分類')
            rating = poi.get('avg_rating', 0)
            
            print(f"🔍 審核第 {rank}/{len(ranked_pois)} 名: {poi_name}")
            print(f"   類別: {poi_category} | 評分: {rating:.1f}⭐")
            
            # LLM審核
            if self.is_travel_relevant(poi, user_categories):
                approved_pois.append(poi)
                print(f"   ✅ 通過審核! (已收集 {len(approved_pois)} 個)")
            else:
                print(f"   ❌ 審核未通過 (不適合旅客)")
            
            print()
            
            # 控制請求頻率
            if rank < len(ranked_pois):
                time.sleep(self.delay_between_requests)
        
        # 最終結果
        final_count = len(approved_pois)
        print(f"\n🏆 最終結果:")
        print(f"   審核完成: {len(ranked_pois)} 個POI")
        print(f"   通過審核: {final_count} 個POI")
        print(f"   返回前 {target_k} 名")
        
        # 返回前K個通過審核的POI
        return approved_pois[:target_k]
    
    def _build_travel_relevance_prompt(self, poi: Dict[str, Any], user_categories: Optional[List[str]] = None) -> str:
        """構建旅遊相關性判斷提示 - 嚴格審核版本"""
        poi_name = poi.get('name', '未知')
        poi_category = poi.get('primary_category', '未分類')
        poi_description = poi.get('description', '')
        stars = poi.get('stars', 0)
        review_count = poi.get('review_count', 0)
        
        # 如果有用戶活動意圖，使用嚴格的匹配邏輯
        if user_categories and len(user_categories) > 0:
            user_intent = user_categories[0]  # 用戶的活動需求（如「喝咖啡」）
            
            prompt = f"""你是一個**非常嚴格**的旅遊推薦審核專家。用戶明確表示想要："{user_intent}"。

請**嚴格審核**以下地點是否**直接符合**用戶的需求：

名稱: {poi_name}
類別: {poi_category}
評分: {stars} 星 ({review_count} 評論)
描述: {poi_description}

**嚴格審核標準**：
1. ✅ 只有當這個地點**主要提供**用戶想要的活動時，才回答 yes
   - 例如：用戶想「喝咖啡」→ 咖啡廳、咖啡館 ✅ | 餐廳、酒吧 ❌
   - 例如：用戶想「吃海鮮」→ 海鮮餐廳 ✅ | 一般餐廳、咖啡廳 ❌
   - 例如：用戶想「看博物館」→ 博物館、美術館 ✅ | 公園、商店 ❌
   - 例如：用戶想「吃義大利菜」→ 義大利餐廳 ✅ | 其他國家料理 ❌

2. ❌ **拒絕**以下情況：
   - 地點類別與用戶需求不直接相關
   - 地點「也許可以」但不是主要用途
   - 評分過低（< 3.0 星）或評論極少（< 5 個）
   - 名稱看起來像住宅、辦公室、停車場等非商業場所

3. 🔍 **特別注意**：
   - 用戶說「喝咖啡」就**只推薦**咖啡廳/咖啡館
   - 不要推薦「也可以喝到咖啡」的餐廳或酒吧
   - 寧可錯殺不可放過，保持高標準
   - 類別名稱必須**直接包含**或**明確相關**於用戶需求

**只回答 yes 或 no。如果不確定，請回答 no。**

答案:"""
        else:
            # 沒有特定需求時，使用一般審核標準
            prompt = f"""你是一個旅遊推薦專家。請判斷以下POI是否適合作為旅遊推薦：

POI資訊:
- 名稱: {poi_name}
- 類別: {poi_category}
- 評分: {stars} 星 ({review_count} 評論)
- 描述: {poi_description}

判斷標準：
✅ 適合旅客的POI:
- 旅遊景點、博物館、公園
- 餐廳、咖啡館、購物中心
- 酒店、民宿等住宿
- 娛樂場所、劇院、遊樂園
- 交通樞紐、機場、車站
- 旅遊服務設施

❌ 不適合旅客的POI:
- 倉儲設施、自助倉庫
- 工業設施、工廠
- 辦公大樓、私人住宅
- 汽車維修、技術服務
- 醫療診所（除非緊急）
- 評分 < 3.0 星或評論 < 5 個

請只回答 yes 或 no。

答案:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """調用LLM API"""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 200,
                "stream": False
            }
            
            response = self.session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return content.strip()
            else:
                print(f"   LLM API錯誤: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   LLM API超時")
            return None
        except Exception as e:
            print(f"   LLM API調用失敗: {e}")
            return None
    
    def _parse_travel_relevance_response(self, response: str) -> bool:
        """解析LLM回應 - 嚴格模式"""
        if not response:
            return False
            
        response_lower = response.lower().strip()
        
        # 優先檢查明確的 yes/no（嚴格模式）
        if response_lower.startswith('yes') or response_lower == 'yes':
            return True
        elif response_lower.startswith('no') or response_lower == 'no':
            return False
        
        # 中文適合/不適合
        if '適合' in response and '不適合' not in response:
            return True
        elif '不適合' in response:
            return False
        
        # 英文關鍵詞（更嚴格）
        if 'yes' in response_lower and 'no' not in response_lower:
            return True
        elif 'no' in response_lower:
            return False
            
        # 其他積極詞
        positive_keywords = ['suitable', 'appropriate', 'relevant', 'recommend']
        negative_keywords = ['not suitable', 'inappropriate', 'irrelevant', 'not recommend', 'reject']
        
        for keyword in negative_keywords:
            if keyword in response_lower:
                return False
                
        for keyword in positive_keywords:
            if keyword in response_lower:
                return True
        
        # 嚴格模式：無法解析則拒絕（寧可錯殺）
        print(f"   ⚠️ 無法解析LLM回應，嚴格模式預設拒絕: {response[:50]}")
        return False
    
    def _fallback_travel_filter(self, poi: Dict[str, Any]) -> bool:
        """備用過濾邏輯 - 當LLM失敗時使用"""
        poi_name = poi.get('name', '').lower()
        poi_category = poi.get('primary_category', '').lower()
        
        # 明確不適合旅客的類別
        non_travel_categories = {
            'self storage', 'storage', 'warehouse',
            'auto repair', 'car repair', 'automotive',
            'office building', 'business service',
            'industrial', 'factory', 'manufacturing',
            'medical clinic', 'dental', 'pharmacy',
            'real estate', 'insurance', 'financial',
            'legal service', 'accounting'
        }
        
        # 明確不適合的關鍵詞
        non_travel_keywords = {
            'storage', 'warehouse', 'repair shop',
            'office', 'clinic', 'dental', 'insurance',
            'real estate', 'law firm', 'accounting'
        }
        
        # 檢查類別
        for non_travel in non_travel_categories:
            if non_travel in poi_category:
                return False
        
        # 檢查名稱
        for keyword in non_travel_keywords:
            if keyword in poi_name:
                return False
        
        # 預設為適合
        return True


# 測試函數
def test_llm_filter():
    """測試LLM過濾器"""
    print("🧪 測試LLM過濾器")
    
    # 創建過濾器
    llm_filter = SimpleLLMFilter()
    
    # 測試POI
    test_pois = [
        {
            'name': '星巴克咖啡',
            'primary_category': 'Cafe',
            'description': '知名連鎖咖啡店',
            'avg_rating': 4.5
        },
        {
            'name': 'Purely Storage',
            'primary_category': 'Self Storage',
            'description': '自助倉儲服務',
            'avg_rating': 4.2
        },
        {
            'name': '現代藝術博物館',
            'primary_category': 'Museum',
            'description': '展示現代藝術作品的博物館',
            'avg_rating': 4.8
        }
    ]
    
    print(f"\n📋 測試POI列表:")
    for i, poi in enumerate(test_pois, 1):
        print(f"  {i}. {poi['name']} ({poi['primary_category']})")
    
    # 測試逐一審核
    print(f"\n🎯 測試逐一審核功能:")
    approved = llm_filter.sequential_llm_filter_top_k(test_pois, target_k=2)
    
    print(f"\n📊 最終結果:")
    for i, poi in enumerate(approved, 1):
        print(f"  {i}. {poi['name']} - ✅ 通過審核")


if __name__ == "__main__":
    test_llm_filter()
