"""
簡化的LLM過濾器
專門用於過濾旅遊推薦中不適合旅客的POI
"""

import requests
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, Tuple

try:
    import aiohttp
    ASYNC_SUPPORTED = True
except ImportError:
    ASYNC_SUPPORTED = False
    print("⚠️ aiohttp 未安裝，併發功能不可用。安裝: pip install aiohttp")


class SimpleLLMFilter:
    """簡化的LLM過濾器 - 專注旅遊相關性"""
    
    def __init__(
        self,
        base_url: str = "140.125.248.15:30020",
        model: str = "meta/llama3-70b-instruct",
        timeout: int = 30,
        delay_between_requests: float = 0.5,
        max_concurrent: int = 5  # 新增：最大併發數
    ):
        self.base_url = f"http://{base_url}" if not base_url.startswith('http') else base_url
        self.model = model
        self.timeout = timeout
        self.delay_between_requests = delay_between_requests
        self.max_concurrent = max_concurrent
        
        # 設置會話
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        print(f"LLM過濾器初始化完成")
        print(f"   端點: {self.base_url}")
        print(f"   模型: {self.model}")
        if ASYNC_SUPPORTED:
            print(f"   併發支援: ✅ (最大 {max_concurrent} 併發)")
        else:
            print(f"   併發支援: ❌ (需安裝 aiohttp)")
    
    def is_travel_relevant(self, poi: Dict[str, Any], user_categories: Optional[List[str]] = None) -> tuple[bool, str, float]:
        """
        判斷POI是否適合旅客，並返回詳細理由
        
        Args:
            poi: POI資訊字典
            user_categories: 用戶偏好的類別列表（可選）
            
        Returns:
            (is_relevant, reason, score): (是否適合, 理由, 評分0-10)
        """
        try:
            # 構建判斷提示
            prompt = self._build_travel_relevance_prompt(poi, user_categories)
            
            # 調用LLM
            response = self._call_llm(prompt)
            
            if response:
                # 解析回應（返回是否通過、理由、評分）
                return self._parse_travel_relevance_response(response)
            
            # 失敗時的預設判斷
            fallback_result = self._fallback_travel_filter(poi)
            return fallback_result, "LLM API 失敗，使用備用規則", 5.0
            
        except Exception as e:
            print(f"   LLM判斷失敗: {e}")
            fallback_result = self._fallback_travel_filter(poi)
            return fallback_result, f"錯誤: {str(e)}", 5.0
    
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
        
        print(f"開始LLM旅遊相關性過濾...")
        print(f"   輸入POI數量: {len(pois)}")
        if user_categories:
            print(f"   用戶偏好類別: {', '.join(user_categories)}")
        
        filtered_pois = []
        rejected_count = 0
        
        for i, poi in enumerate(pois, 1):
            poi_name = poi.get('name', '未知POI')
            poi_category = poi.get('primary_category', '未分類')
            
            print(f"   ({i}/{len(pois)}) 檢查: {poi_name} ({poi_category})")
            
            is_relevant, reason, score = self.is_travel_relevant(poi, user_categories)
            
            if is_relevant:
                filtered_pois.append(poi)
                print(f"     ACCEPT - 評分: {score:.1f}/10")
                print(f"     REASON: {reason}")
            else:
                rejected_count += 1
                print(f"     REJECTED - 評分: {score:.1f}/10")
                print(f"     REASON: {reason}")
            
            # 控制請求頻率
            if i < len(pois):  # 不是最後一個
                time.sleep(self.delay_between_requests)
        
        print(f"LLM過濾完成!")
        print(f"ACCEPT: {len(filtered_pois)} 個")
        print(f"REJECTED: {rejected_count} 個")
        
        return filtered_pois
    
    def _filter_by_bounding_box(
        self,
        pois: List[Dict[str, Any]],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """
        使用起終點構建矩形邊界框，過濾掉範圍外的 POI
        
        Args:
            pois: POI 列表
            start_location: 起點 (latitude, longitude)
            end_location: 終點 (latitude, longitude)
            
        Returns:
            在矩形邊界框內的 POI 列表
        """
        start_lat, start_lng = start_location
        end_lat, end_lng = end_location
        
        # 計算矩形邊界（對角線兩點）
        min_lat = min(start_lat, end_lat)
        max_lat = max(start_lat, end_lat)
        min_lng = min(start_lng, end_lng)
        max_lng = max(start_lng, end_lng)
        
        print(f"\n📦 地理邊界框過濾:")
        print(f"   起點: ({start_lat:.6f}, {start_lng:.6f})")
        print(f"   終點: ({end_lat:.6f}, {end_lng:.6f})")
        print(f"   邊界框: 緯度 [{min_lat:.6f}, {max_lat:.6f}]")
        print(f"           經度 [{min_lng:.6f}, {max_lng:.6f}]")
        
        # 過濾 POI
        filtered_pois = []
        for poi in pois:
            lat = poi.get('latitude', 0)
            lng = poi.get('longitude', 0)
            
            # 檢查是否在矩形範圍內
            if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
                filtered_pois.append(poi)
        
        print(f"   輸入 POI: {len(pois)} 個")
        print(f"   矩形內 POI: {len(filtered_pois)} 個")
        print(f"   過濾掉: {len(pois) - len(filtered_pois)} 個 ({100 * (len(pois) - len(filtered_pois)) / len(pois) if pois else 0:.1f}%)")
        
        return filtered_pois
    
    def sequential_llm_filter_top_k(
        self, 
        ranked_pois: List[Dict[str, Any]], 
        target_k: int,
        start_location: Optional[Tuple[float, float]] = None,
        end_location: Optional[Tuple[float, float]] = None,
        multiplier: int = 3,
        user_categories: Optional[List[str]] = None,
        early_stop: bool = True,
        early_stop_buffer: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        按排序逐一審核，直到收集到target_k個通過的POI（支持早停）
        
        這是您要求的核心功能：
        1. 【新增】地理邊界框預過濾（如提供起終點）
        2. 從第1名開始逐一審核
        3. 通過LLM審核的加入最終列表
        4. 收集到足夠多的候選後早停（可配置）
        
        Args:
            ranked_pois: 已排序的POI列表
            target_k: 目標數量
            start_location: 起點座標 (latitude, longitude)，可選
            end_location: 終點座標 (latitude, longitude)，可選
            multiplier: 初始搜索倍數（搜索前 target_k * multiplier 個）
            user_categories: 用戶偏好的類別列表（可選）
            early_stop: 是否啟用早停（默認True）
            early_stop_buffer: 早停緩衝倍數（默認1.5，即收集到 target_k * 1.5 個後停止）
            
        Returns:
            通過LLM審核的TOP K POI列表
        """
        if not ranked_pois:
            return []
        
        # 【新增】地理邊界框預過濾
        if start_location and end_location:
            print(f"\n🌍 啟用地理邊界框預過濾")
            ranked_pois = self._filter_by_bounding_box(
                ranked_pois, 
                start_location, 
                end_location
            )
            
            if not ranked_pois:
                print("⚠️ 警告: 地理過濾後沒有剩餘 POI")
                return []
        
        # 計算早停閾值
        early_stop_threshold = int(target_k * early_stop_buffer) if early_stop else float('inf')
        
        print(f"\n開始逐一LLM審核流程")
        print(f"   目標: TOP {target_k} 推薦")
        print(f"   輸入: {len(ranked_pois)} 個排序POI")
        if early_stop:
            print(f"   早停策略: 收集到 {early_stop_threshold} 個候選後停止 ({early_stop_buffer}x buffer)")
        else:
            print(f"   審核範圍: 全部 {len(ranked_pois)} 個候選（不早停）")
        if user_categories:
            print(f"   用戶偏好類別: {', '.join(user_categories)}")
        
        approved_pois = []
        search_limit = min(len(ranked_pois), target_k * multiplier)
        
        print()
        
        # 從第1名開始逐一審核 - 支持早停
        for rank, poi in enumerate(ranked_pois, 1):
            # 早停檢查：收集到足夠多的候選後停止
            if early_stop and len(approved_pois) >= early_stop_threshold:
                print(f"\n早停觸發！")
                print(f"   已收集 {len(approved_pois)} 個候選（目標 {target_k} 個）")
                print(f"   停止審核，節省 {len(ranked_pois) - rank + 1} 次 LLM 調用")
                break
            
            poi_name = poi.get('name', '未知POI')
            poi_category = poi.get('primary_category', '未分類')
            rating = poi.get('avg_rating', 0)
            
            print(f"審核第 {rank}/{len(ranked_pois)} 名: {poi_name}")
            print(f"   類別: {poi_category} | 評分: {rating:.1f}⭐")
            
            # LLM審核（獲取詳細理由）
            is_relevant, reason, llm_score = self.is_travel_relevant(poi, user_categories)
            
            print(f"  LLM評分: {llm_score:.1f}/10")
            print(f"  理由: {reason}")
            
            if is_relevant:
                approved_pois.append(poi)
                print(f"ACCEPT (已收集 {len(approved_pois)}/{early_stop_threshold if early_stop else 'inf'} 個)")
            else:
                print(f"REJECTED")
            
            print()
            
            # 控制請求頻率
            if rank < len(ranked_pois):
                time.sleep(self.delay_between_requests)
        
        # 最終結果
        final_count = len(approved_pois)
        print(f"\n最終結果:")
        print(f"   審核完成: {rank} 個POI")
        print(f"   通過審核: {final_count} 個POI")
        print(f"   返回前 {target_k} 名")
        if early_stop and final_count > target_k:
            print(f"   節省時間: 跳過 {len(ranked_pois) - rank} 次審核")
        
        # 返回前K個通過審核的POI
        return approved_pois[:target_k]
    
    def sequential_llm_filter_top_k_concurrent(
        self, 
        ranked_pois: List[Dict[str, Any]], 
        target_k: int,
        start_location: Optional[Tuple[float, float]] = None,
        end_location: Optional[Tuple[float, float]] = None,
        batch_size: int = 10,
        user_categories: Optional[List[str]] = None,
        early_stop: bool = True,
        early_stop_buffer: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        併發批量審核版本 - 顯著提升速度
        
        使用批量併發 LLM 調用，可將審核時間縮短 5-10 倍
        
        Args:
            ranked_pois: 已排序的POI列表
            target_k: 目標數量
            start_location: 起點座標 (latitude, longitude)，可選
            end_location: 終點座標 (latitude, longitude)，可選
            batch_size: 每批次併發數量（默認10）
            user_categories: 用戶偏好的類別列表（可選）
            early_stop: 是否啟用早停（默認True）
            early_stop_buffer: 早停緩衝倍數（默認1.5）
            
        Returns:
            通過LLM審核的TOP K POI列表
        """
        if not ASYNC_SUPPORTED:
            print("⚠️ 併發功能不可用，降級到順序處理")
            return self.sequential_llm_filter_top_k(
                ranked_pois, target_k, start_location, end_location,
                3, user_categories, early_stop, early_stop_buffer
            )
        
        if not ranked_pois:
            return []
        
        # 地理邊界框預過濾
        if start_location and end_location:
            print(f"\n🌍 啟用地理邊界框預過濾")
            ranked_pois = self._filter_by_bounding_box(
                ranked_pois, 
                start_location, 
                end_location
            )
            
            if not ranked_pois:
                print("⚠️ 警告: 地理過濾後沒有剩餘 POI")
                return []
        
        early_stop_threshold = int(target_k * early_stop_buffer) if early_stop else float('inf')
        
        print(f"\n🚀 開始併發LLM審核流程")
        print(f"   目標: TOP {target_k} 推薦")
        print(f"   輸入: {len(ranked_pois)} 個排序POI")
        print(f"   併發批次大小: {batch_size}")
        if early_stop:
            print(f"   早停策略: 收集到 {early_stop_threshold} 個候選後停止")
        if user_categories:
            print(f"   用戶偏好類別: {', '.join(user_categories)}")
        
        approved_pois = []
        processed_count = 0
        
        # 分批併發處理
        for batch_start in range(0, len(ranked_pois), batch_size):
            # 早停檢查
            if early_stop and len(approved_pois) >= early_stop_threshold:
                print(f"\n✋ 早停觸發！")
                print(f"   已收集 {len(approved_pois)} 個候選（目標 {target_k} 個）")
                print(f"   停止審核，節省 {len(ranked_pois) - processed_count} 次 LLM 調用")
                break
            
            batch_end = min(batch_start + batch_size, len(ranked_pois))
            batch_pois = ranked_pois[batch_start:batch_end]
            
            print(f"\n📦 批次 {batch_start//batch_size + 1}: 併發處理 {len(batch_pois)} 個 POI...")
            
            # 併發調用 LLM
            import asyncio
            batch_results = asyncio.run(
                self._batch_call_llm_async(batch_pois, user_categories)
            )
            
            # 處理結果
            for poi, (is_relevant, reason, llm_score) in zip(batch_pois, batch_results):
                processed_count += 1
                poi_name = poi.get('name', '未知POI')
                poi_category = poi.get('primary_category', '未分類')
                
                print(f"   [{processed_count}/{len(ranked_pois)}] {poi_name}")
                print(f"       評分: {llm_score:.1f}/10 | {poi_category}")
                
                if is_relevant:
                    approved_pois.append(poi)
                    print(f"       ✅ ACCEPT (已收集 {len(approved_pois)} 個)")
                else:
                    print(f"       ❌ REJECT")
        
        # 最終結果
        print(f"\n✨ 併發審核完成!")
        print(f"   審核完成: {processed_count} 個POI")
        print(f"   通過審核: {len(approved_pois)} 個POI")
        print(f"   返回前 {target_k} 名")
        
        return approved_pois[:target_k]
    
    def _build_travel_relevance_prompt(self, poi: Dict[str, Any], user_categories: Optional[List[str]] = None) -> str:
        """構建旅遊相關性判斷提示 - 嚴格審核版本"""
        poi_name = poi.get('name', '未知')
        poi_category = poi.get('primary_category', '未分類')
        poi_description = poi.get('description', '')
        # 使用正確的欄位名稱
        stars = poi.get('avg_rating', poi.get('stars', 0))
        review_count = poi.get('num_reviews', poi.get('review_count', 0))
        
        # 如果有用戶活動意圖，使用嚴格的匹配邏輯
        if user_categories and len(user_categories) > 0:
            user_intent = user_categories[0]  # 用戶的活動需求（如「喝咖啡」）
            
            prompt = f"""You are a **very strict** travel recommendation auditor. The user explicitly states they want: "{user_intent}".

Please **strictly evaluate** whether the following place **directly satisfies** the user's request:

Name: {poi_name}
Category: {poi_category}
Rating: {stars} stars ({review_count} reviews)
Description: {poi_description}

**Strict evaluation criteria**:
1. Only accept when the place **primarily offers** the activity the user wants.
   - Example: user wants “drink coffee” → cafe, coffee shop ACCEPT | restaurant, bar REJECT
   - Example: user wants “eat seafood” → seafood restaurant ACCEPT | general restaurant, coffee shop REJECT
   - Example: user wants “visit a museum” → museum, art gallery ACCEPT | park, shop REJECT
   - Example: user wants “eat Italian” → Italian restaurant ACCEPT | other cuisines REJECT

2. **Automatically REJECT** in the following cases:
   - The place’s category is not directly related to the user’s need.
   - The place “might work” but it is not the primary purpose.
   - Rating is too low (< 3.0 stars) or there are very few reviews (< 5).
   - The name appears to be a residence, office, parking lot, or other non-commercial site.

3. **Special notes**:
   - The category name must **directly include** or be **clearly relevant** to the user’s need.
   - Better to wrongly reject than to wrongly accept—maintain a high standard.

**Reply using the exact format below (you must follow this format strictly):**

Decision: [ACCEPT or REJECT]
Score: [a number 0–10 indicating how well it matches the user’s need]
Reason: [detailed explanation why you accepted or rejected, analyzing the place’s category in relation to the user’s need]

Example answer:
Decision: REJECT
Score: 3
Reason: Although this is a restaurant that may serve coffee, the user explicitly wants to “drink coffee” and therefore a professional coffee shop should be recommended rather than a general restaurant. Category mismatch.

Now please evaluate:"""
        else:
            # 沒有特定需求時，使用一般審核標準
            prompt = f"""You are a travel recommendation expert. Please determine whether the following POI is suitable for travel recommendation:

POI Information:
- Name: {poi_name}
- Category: {poi_category}
- Rating: {stars} stars ({review_count} reviews)
- Description: {poi_description}

Evaluation Criteria:
Suitable POIs for travelers:
- Tourist attractions, museums, parks
- Restaurants, cafes, shopping centers
- Hotels, inns, and other accommodations
- Entertainment venues, theaters, amusement parks
- Transportation hubs, airports, train/bus stations
- Tourism service facilities

Unsuitable POIs for travelers:
- Storage facilities, self-storage units
- Industrial sites, factories
- Office buildings, private residences
- Auto repair shops, technical services
- Medical clinics (except for emergencies)
- Rating < 3.0 stars or reviews < 5

**Please respond in the following format:**

Decision: [ACCEPT or REJECT]
Score: [a number from 0–10 indicating how suitable this POI is for travel recommendation]
Reason: [detailed explanation of why it is or isn’t suitable as a travel recommendation]

Now please evaluate:"""
        
        return prompt
    
    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> Optional[str]:
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
                "temperature": temperature,
                "max_tokens": max_tokens,
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
    
    async def _call_llm_async(self, session: 'aiohttp.ClientSession', prompt: str) -> Optional[str]:
        """異步調用LLM API"""
        if not ASYNC_SUPPORTED:
            return None
            
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
                "max_tokens": 500,
                "stream": False
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return content.strip()
                else:
                    return None
                    
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            return None
    
    async def _batch_call_llm_async(
        self, 
        pois: List[Dict[str, Any]], 
        user_categories: Optional[List[str]] = None
    ) -> List[Tuple[bool, str, float]]:
        """批量異步調用LLM API
        
        Args:
            pois: POI 列表
            user_categories: 用戶偏好類別
            
        Returns:
            List of (is_relevant, reason, score) tuples
        """
        if not ASYNC_SUPPORTED:
            # 降級到同步調用
            results = []
            for poi in pois:
                result = self.is_travel_relevant(poi, user_categories)
                results.append(result)
            return results
        
        async with aiohttp.ClientSession() as session:
            # 創建信號量控制併發數
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_poi(poi: Dict[str, Any]) -> Tuple[bool, str, float]:
                async with semaphore:
                    try:
                        prompt = self._build_travel_relevance_prompt(poi, user_categories)
                        response = await self._call_llm_async(session, prompt)
                        
                        if response:
                            return self._parse_travel_relevance_response(response)
                        else:
                            # API 失敗，使用備用規則
                            fallback_result = self._fallback_travel_filter(poi)
                            return fallback_result, "LLM API 失敗，使用備用規則", 5.0
                    except Exception as e:
                        fallback_result = self._fallback_travel_filter(poi)
                        return fallback_result, f"錯誤: {str(e)}", 5.0
            
            # 併發執行所有請求
            tasks = [process_poi(poi) for poi in pois]
            results = await asyncio.gather(*tasks)
            
            return results
    
    def _parse_travel_relevance_response(self, response: str) -> tuple[bool, str, float]:
        """解析LLM回應 - 結構化解析
        
        Returns:
            (is_accepted, reason, score): (是否通過, 理由, 評分0-10)
        """
        if not response:
            return False, "LLM 無回應", 0.0
        
        try:
            # 解析結構化回應
            lines = response.strip().split('\n')
            decision = None
            score = 5.0
            reason = "無法解析理由"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析決策
                if line.startswith('決策:') or line.startswith('Decision:'):
                    decision_text = line.split(':', 1)[1].strip().upper()
                    if 'ACCEPT' in decision_text or '通過' in decision_text:
                        decision = True
                    elif 'REJECT' in decision_text or '拒絕' in decision_text:
                        decision = False
                
                # 解析評分
                elif line.startswith('評分:') or line.startswith('Score:'):
                    score_text = line.split(':', 1)[1].strip()
                    # 提取數字
                    import re
                    score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if score_match:
                        score = float(score_match.group(1))
                        # 確保在 0-10 範圍內
                        score = max(0.0, min(10.0, score))
                
                # 解析理由
                elif line.startswith('理由:') or line.startswith('Reason:'):
                    reason = line.split(':', 1)[1].strip()
            
            # 如果無法解析決策，嘗試從回應文字判斷
            if decision is None:
                response_lower = response.lower()
                if 'accept' in response_lower or '通過' in response:
                    decision = True
                    if reason == "無法解析理由":
                        reason = "根據回應文字判斷為通過"
                elif 'reject' in response_lower or '拒絕' in response:
                    decision = False
                    if reason == "無法解析理由":
                        reason = "根據回應文字判斷為拒絕"
                else:
                    # 無法判斷，嚴格模式預設拒絕
                    decision = False
                    reason = f"無法解析決策，預設拒絕。原始回應: {response[:100]}"
                    print(f"WARNING: 無法解析結構化回應: {response[:100]}")
            
            return decision, reason, score
            
        except Exception as e:
            print(f"WARNING: 解析LLM回應時出錯: {e}")
            return False, f"解析錯誤: {str(e)}", 0.0
    
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
    
    def generate_itinerary(
        self,
        pois: List[Dict[str, Any]],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        activity_intent: str = "旅遊探索",
        time_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用 LLM 將 POI 列表組合成合理的旅遊行程
        
        Args:
            pois: POI 列表（已經過 reranking）
            start_location: 起點座標 (lat, lon)
            end_location: 終點座標 (lat, lon)
            activity_intent: 活動意圖/需求
            time_budget: 時間預算（分鐘），可選
        
        Returns:
            {
                'itinerary': [
                    {
                        'order': 1,
                        'poi': {...},
                        'reason': '選擇理由',
                        'estimated_duration': 60,  # 建議停留時間（分鐘）
                    },
                    ...
                ],
                'total_duration': 180,  # 總時間（分鐘）
                'total_distance': 15.5,  # 總距離（公里）
                'summary': '行程摘要說明',
                'tips': ['建議1', '建議2', ...]
            }
        """
        try:
            # 構建行程規劃 prompt
            prompt = self._build_itinerary_prompt(
                pois, start_location, end_location, activity_intent, time_budget
            )
            
            # 調用 LLM
            response = self._call_llm(prompt, temperature=0.7, max_tokens=2000)
            
            if response:
                # 解析行程
                return self._parse_itinerary_response(response, pois)
            else:
                # 失敗時返回簡單序列
                return self._fallback_itinerary(pois)
                
        except Exception as e:
            print(f"⚠️ LLM 行程生成失敗: {e}")
            return self._fallback_itinerary(pois)
    
    def _build_itinerary_prompt(
        self,
        pois: List[Dict[str, Any]],
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        activity_intent: str,
        time_budget: Optional[int]
    ) -> str:
        """構建行程規劃 prompt"""
        
        # POI 信息
        poi_info = []
        for idx, poi in enumerate(pois, 1):
            info = f"{idx}. {poi.get('name', 'Unknown')}"
            info += f" ({poi.get('primary_category', poi.get('category', 'N/A'))})"
            
            if 'avg_rating' in poi:
                info += f" - 評分: {poi['avg_rating']:.1f}⭐"
            
            if 'detour_info' in poi and poi['detour_info']:
                extra_time = poi['detour_info'].get('extra_duration', 0) / 60.0
                info += f" - 繞道: +{extra_time:.0f}分鐘"
            
            if 'score' in poi:
                info += f" - 推薦分數: {poi['score']:.2f}"
            
            poi_info.append(info)
        
        poi_list_str = "\n".join(poi_info)
        
        time_constraint = f"\n時間預算: {time_budget} 分鐘" if time_budget else ""
        
        prompt = f"""你是一位專業的旅遊行程規劃師。請根據以下信息規劃一個合理的旅遊行程。

起點座標: {start_location[0]:.6f}, {start_location[1]:.6f}
終點座標: {end_location[0]:.6f}, {end_location[1]:.6f}
旅遊需求: {activity_intent}{time_constraint}

候選景點（已按推薦程度排序）:
{poi_list_str}

請規劃一個**合理的旅遊行程**，考慮：
1. 地理位置順序（避免來回繞路）
2. 景點類型搭配（豐富度和多樣性）
3. 時間分配（每個景點建議停留時間）
4. 整體路線流暢性

請以以下 JSON 格式回覆（**只回覆 JSON，不要其他文字**）:
{{
  "selected_pois": [
    {{
      "poi_index": 1,
      "order": 1,
      "reason": "選擇理由",
      "estimated_duration_minutes": 60
    }}
  ],
  "summary": "整體行程摘要說明",
  "tips": ["建議1", "建議2"]
}}

注意：
- poi_index 是原始列表的編號（1-based）
- order 是行程中的順序（1-based）
- 選擇 3-7 個景點為佳
- 確保路線合理，避免過度繞路"""

        return prompt
    
    def _parse_itinerary_response(
        self,
        response: str,
        pois: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """解析 LLM 的行程回覆"""
        try:
            # 嘗試提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group()
            
            data = json.loads(response)
            
            # 構建行程
            itinerary = []
            total_duration = 0
            
            for item in data.get('selected_pois', []):
                poi_idx = item.get('poi_index', 0) - 1  # 轉為 0-based
                
                if 0 <= poi_idx < len(pois):
                    poi = pois[poi_idx]
                    duration = item.get('estimated_duration_minutes', 60)
                    
                    # 調試：確認 POI 有座標
                    if 'latitude' not in poi or 'longitude' not in poi:
                        print(f"⚠️ LLM選中的POI缺少座標: {poi.get('name', 'Unknown')}")
                        print(f"   POI keys: {list(poi.keys())}")
                    
                    itinerary.append({
                        'order': item.get('order', len(itinerary) + 1),
                        'poi': poi,
                        'reason': item.get('reason', ''),
                        'estimated_duration': duration
                    })
                    
                    total_duration += duration
            
            # 計算總距離（概估）
            total_distance = 0.0
            for poi in itinerary:
                if 'detour_info' in poi['poi'] and poi['poi']['detour_info']:
                    extra_dist = poi['poi']['detour_info'].get('extra_distance', 0) / 1000.0
                    total_distance += extra_dist
            
            return {
                'itinerary': itinerary,
                'total_duration': total_duration,
                'total_distance': total_distance,
                'summary': data.get('summary', '精彩的旅遊行程'),
                'tips': data.get('tips', [])
            }
            
        except Exception as e:
            print(f"⚠️ 解析行程回覆失敗: {e}")
            print(f"   原始回覆: {response[:200]}")
            return self._fallback_itinerary(pois)
    
    def _fallback_itinerary(self, pois: List[Dict[str, Any]]) -> Dict[str, Any]:
        """備用行程生成（簡單按順序）"""
        itinerary = []
        total_duration = 0
        
        # 最多選擇前 5 個
        for idx, poi in enumerate(pois[:5], 1):
            duration = 60  # 預設 60 分鐘
            itinerary.append({
                'order': idx,
                'poi': poi,
                'reason': '推薦景點',
                'estimated_duration': duration
            })
            total_duration += duration
        
        return {
            'itinerary': itinerary,
            'total_duration': total_duration,
            'total_distance': 0.0,
            'summary': '按推薦順序安排的行程',
            'tips': ['這是備用行程安排']
        }


# 測試函數
def test_llm_filter():
    """測試LLM過濾器"""
    print("測試LLM過濾器")
    
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
    
    print(f"\n測試POI列表:")
    for i, poi in enumerate(test_pois, 1):
        print(f"  {i}. {poi['name']} ({poi['primary_category']})")
    
    # 測試逐一審核
    print(f"\n測試逐一審核功能:")
    approved = llm_filter.sequential_llm_filter_top_k(test_pois, target_k=2)
    
    print(f"\n最終結果:")
    for i, poi in enumerate(approved, 1):
        print(f"  {i}. {poi['name']} - 通過審核")


if __name__ == "__main__":
    test_llm_filter()
