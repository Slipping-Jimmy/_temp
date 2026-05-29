import json
import time
import csv
import random
from locust import HttpUser, TaskSet, task, between, events

BASE_URL = "http://192.168.8.134:25207"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/v1/chat/completions"

MODEL_NAME = "chatbot_lora"

TOKEN_TIME_CSV = "2gpu_1p_token_times.csv"
SUMMARY_CSV = "vllm_summary.csv"

with open("questions_nhi.log", "r", encoding="utf-8") as f:
    user_questions = [q for q in f.read().splitlines() if q.strip()]

first_token_times = []
token_30_times = []
token_300_times = []
token_3000_times = []
final_token_times = []


def percentile(values, p):
    if not values:
        return "N/A"
    values = sorted(values)
    k = int((len(values) - 1) * p / 100)
    return values[k]


class UserBehavior(TaskSet):

    @task
    def post_chat_completion(self):
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": MODEL_NAME,
            "temperature": 0,
            "top_p": 0.1,
            "stream": True,
            "max_tokens": 3000,
            "messages": [
                {"role": "user", "content": random.choice(user_questions)}
            ],
        }

        start_time = time.perf_counter()
        first_token_time = None
        final_token_time = None
        token_count = 0

        with self.client.post(
            CHAT_COMPLETIONS_URL,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            stream=True,
            catch_response=True,
            name="/v1/chat/completions",
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

                    data = output[5:].strip()

                    if data == "[DONE]":
                        final_token_time = time.perf_counter() - start_time
                        final_token_times.append(final_token_time)
                        response.success()
                        return

                    json_data = json.loads(data)
                    choice = json_data["choices"][0]
                    content = choice.get("delta", {}).get("content")

                    if content:
                        token_count += 1

                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start_time
                            first_token_times.append(first_token_time)
                            print(f"First token return time: {first_token_time:.6f} seconds")

                        if token_count == 30:
                            token_30_times.append(time.perf_counter() - start_time)
                        elif token_count == 300:
                            token_300_times.append(time.perf_counter() - start_time)
                        elif token_count == 3000:
                            token_3000_times.append(time.perf_counter() - start_time)

                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        final_token_time = time.perf_counter() - start_time
                        final_token_times.append(final_token_time)
                        response.success()
                        return

                final_token_time = time.perf_counter() - start_time
                final_token_times.append(final_token_time)
                response.success()

            except Exception as e:
                response.failure(f"Stream parse failed: {e}")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(5, 10)
    host = BASE_URL


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    max_len = max(
        len(first_token_times),
        len(token_30_times),
        len(token_300_times),
        len(token_3000_times),
        len(final_token_times),
        0,
    )

    with open(TOKEN_TIME_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "First Token Time",
            "30 Tokens Time",
            "300 Tokens Time",
            "3000 Tokens Time",
            "Final Token Time",
        ])

        for i in range(max_len):
            writer.writerow([
                first_token_times[i] if i < len(first_token_times) else "N/A",
                token_30_times[i] if i < len(token_30_times) else "N/A",
                token_300_times[i] if i < len(token_300_times) else "N/A",
                token_3000_times[i] if i < len(token_3000_times) else "N/A",
                final_token_times[i] if i < len(final_token_times) else "N/A",
            ])

    count = len(first_token_times)
    avg_ttft = sum(first_token_times) / count if count else "N/A"
    under_3s_count = sum(1 for x in first_token_times if x <= 3)
    under_3s_rate = under_3s_count / count if count else 0

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
            count,
            avg_ttft,
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
