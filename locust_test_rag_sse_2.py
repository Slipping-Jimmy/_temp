import uuid
import time
import csv
import random
import requests
from locust import HttpUser, TaskSet, task, between, events

BASE_URL = "http://192.168.8.134:22999"
CONVERSATION_URL = f"{BASE_URL}/api/chatbot/conversations"
SSE_URL = f"{BASE_URL}/api/chatbot/conversations/{{conversation_id}}/sse"

TOKEN_TIME_CSV = "2gpu_1p_token_times.csv"
SUMMARY_CSV = "rag_sse_summary.csv"

headers = {"Content-Type": "application/json"}

with open("questions_nhi.log", "r", encoding="utf-8") as f:
    user_questions = [q for q in f.read().splitlines() if q.strip()]

first_token_times = []
final_token_times = []
chunk_counts = []


def percentile(values, p):
    if not values:
        return "N/A"
    values = sorted(values)
    k = int((len(values) - 1) * p / 100)
    return values[k]


def get_conversation_id(client_user_id: str):
    payload = {
        "user_id": "",
        "_id": client_user_id,
        "from_source": "web_service",
    }

    try:
        response = requests.post(CONVERSATION_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("get_conversation_id failed:", e)
        return None


class UserBehavior(TaskSet):

    @task
    def post_chat_completion(self):
        client_user_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())
        question = random.choice(user_questions)

        conversation_id = get_conversation_id(client_user_id)
        if not conversation_id:
            print("conversation_id is empty, skip this request")
            return

        data = {
            "chat_id": chat_id,
            "_id": client_user_id,
            "user_id": "",
            "from_source": "web_service",
            "question": question,
            "status": "common",
        }

        new_url = SSE_URL.format_map({"conversation_id": conversation_id})

        first_token_time = None
        final_token_time = None
        token_count = 0
        start_time = time.perf_counter()

        with self.client.post(
            new_url,
            headers=headers,
            json=data,
            stream=True,
            catch_response=True,
            name="/api/chatbot/conversations/{conversation_id}/sse",
        ) as response:

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:300]}")
                return

            try:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    output = line.strip()

                    if not output.startswith("data:"):
                        continue

                    token_content = output[5:].strip()

                    if token_content in ("[END]", "[DONE]"):
                        final_token_time = time.perf_counter() - start_time
                        final_token_times.append(final_token_time)
                        chunk_counts.append(token_count)
                        response.success()
                        return

                    if token_content:
                        token_count += 1

                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_time
                        first_token_times.append(first_token_time)
                        print(f"First token return time: {first_token_time:.6f} seconds")

                final_token_time = time.perf_counter() - start_time
                final_token_times.append(final_token_time)
                chunk_counts.append(token_count)
                response.success()

            except Exception as e:
                response.failure(f"SSE parse failed: {e}")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(5, 10)
    host = BASE_URL


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    with open(TOKEN_TIME_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "First Token Time",
            "Final Token Time",
            "Chunk Count",
        ])

        max_len = max(
            len(first_token_times),
            len(final_token_times),
            len(chunk_counts),
            0,
        )

        for i in range(max_len):
            writer.writerow([
                first_token_times[i] if i < len(first_token_times) else "N/A",
                final_token_times[i] if i < len(final_token_times) else "N/A",
                chunk_counts[i] if i < len(chunk_counts) else "N/A",
            ])

    ttft_count = len(first_token_times)
    ttft_avg = sum(first_token_times) / ttft_count if ttft_count else "N/A"
    under_3s_count = sum(1 for x in first_token_times if x <= 3)
    under_3s_rate = under_3s_count / ttft_count if ttft_count else 0

    with open(SUMMARY_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "count",
            "avg_ttft",
            "p50_ttft",
            "p90_ttft",
            "p95_ttft",
            "p99_ttft",
            "max_ttft",
            "under_3s_count",
            "under_3s_rate",
        ])

        writer.writerow([
            ttft_count,
            ttft_avg,
            percentile(first_token_times, 50),
            percentile(first_token_times, 90),
            percentile(first_token_times, 95),
            percentile(first_token_times, 99),
            max(first_token_times) if first_token_times else "N/A",
            under_3s_count,
            under_3s_rate,
        ])

    print(f"Saved token timing report to {TOKEN_TIME_CSV}")
    print(f"Saved summary report to {SUMMARY_CSV}")
