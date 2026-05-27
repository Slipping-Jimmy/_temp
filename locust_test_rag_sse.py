import json
import uuid
import time
import csv
import random
import requests
from locust import HttpUser, TaskSet, task, between, events

# 建立對話的 API URL
url = "http://192.168.8.134:22999/api/chatbot/conversations"

def get_conversation_id():
    payload = {
        "user_id": "",
        "_id": "abcd12345",
        "from_source": "web_service",
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        conversation_id = response.json()
        return conversation_id
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

# 串流端點的 URL (SSE endpoint)
urls = [
    "http://192.168.8.134:22999/api/chatbot/conversations/{conversation_id}/sse",
]

headers = {
    "Content-Type": "application/json"
}

# 讀取測試問題
with open("questions_nhi.log", "r") as f:
    user_questions = f.read().splitlines()

# 存放時間指標的清單
first_token_times = []
token_30_times = []
token_300_times = []
token_3000_times = []
final_token_times = []

class UserBehavior(TaskSet):

    @task
    def post_chat_completion(self):
        data = {
            "chat_id": str(uuid.uuid4()),
            "question": random.choice(user_questions),
            "status": "common",
        }
        
        first_token_time = None
        token_count = 0
        
        target_url = random.choice(urls)
        conversation_id = get_conversation_id()
        new_url = target_url.format_map({"conversation_id": conversation_id})
        
        start_time = time.perf_counter()
        
        # 發送 POST 請求至 SSE 端點並開啟 stream=True
        with self.client.post(new_url, headers=headers, json=data, stream=True, catch_response=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                        
                    output = line.strip()
                    
                    # --- SSE 解析邏輯開始 ---
                    # 確保只處理 "data:" 開頭的行
                    if output.startswith("data:"):
                        
                        # 擷取 "data:" 後面的實際字元 (去掉前 5 個字元並去除頭尾空白)
                        token_content = output[5:].strip()
                        
                        # 判斷是否為結束標記
                        if token_content == "[END]":
                            last_token_time = time.time() - start_time
                            final_token_times.append(last_token_time)
                            break
                        
                        # 若不是結束標記且有內容，則 Token 數量 +1
                        if token_content:
                            token_count += 1
                            
                        # 記錄首字時間
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start_time
                            print(f"First token return time: {first_token_time} seconds")
                            first_token_times.append(first_token_time)
                            
                        # 記錄各階段 Token 時間
                        if token_count == 30:
                            token_30_time = time.time() - start_time
                            token_30_times.append(token_30_time)
                        elif token_count == 300:
                            token_300_time = time.time() - start_time
                            token_300_times.append(token_300_time)
                        elif token_count == 3000:
                            token_3000_time = time.time() - start_time
                            token_3000_times.append(token_3000_time)
                    # --- SSE 解析邏輯結束 ---
                    response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)
    host = url

# 當測試結束時觸發，匯出 CSV
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    with open("1gpu_1p_token_times.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["First Token Time", "30 Tokens Time", "300 Tokens Time", "3000 Tokens Time", "Final Token Time"])
        
        max_len = max(len(first_token_times), len(token_30_times), len(token_300_times), len(token_3000_times), len(final_token_times))
        for i in range(max_len):
            first = first_token_times[i] if i < len(first_token_times) else "N/A"
            token_30 = token_30_times[i] if i < len(token_30_times) else "N/A"
            token_300 = token_300_times[i] if i < len(token_300_times) else "N/A"
            token_3000 = token_3000_times[i] if i < len(token_3000_times) else "N/A"
            final = final_token_times[i] if i < len(final_token_times) else "N/A"
            
            writer.writerow([first, token_30, token_300, token_3000, final])
