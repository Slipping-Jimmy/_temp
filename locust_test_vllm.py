import json
import time
import csv
import random

from locust import HttpUser, TaskSet, task, between, events, runners

# Define two different hosts
hosts = [
    "https://nhismartchattest.intra.nhi.gov.tw/api/vllm",
]


# Questions
with open("questions_nhi.log", "r") as f:
    user_questions = f.read().splitlines()


# Lists to record all first token and final token times
first_token_times = []
token_30_times = []
token_300_times = []
token_3000_times = []
final_token_times = []


class UserBehavior(TaskSet):
    
    @task
    def post_chat_completion(self):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "/workspace/llm_model",
            "temperature": 0,
            "stream": True,
            "max_tokens": 3000,  # Ensure 3,100 tokens are generated
            "messages": [
                {"role": "user", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "ok!"},
                {"role": "user", "content": f"{random.choice(user_questions)}"},
                # {"role": "user", "content": f"{random.choice(user_questions)}.\n!NOTICE!: please generate 5 paragraphs and any paragraph have to be filled in 1,000 tokens. 最後，使用英文回答結束後，請再次使用『中文』、『日文』、『韓文』各自翻譯一次，我總共要看到五種語言的回答哦！", },
            ]
        }

        # Randomly select a host from the list
        selected_host = random.choice(hosts)
        url = f"{selected_host}/v1/chat/completions"

        start_time = time.time()  # Record the start time of the request
        first_token_time = None   # Record the time of the first token response
        token_count = 0
        
        # Send the POST request and handle the streaming response
        with self.client.post(url, headers=headers, data=json.dumps(payload), stream=True, catch_response=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if not line:
                        continue

                    output = line.decode("utf-8")

                    try:
                        # Parse the JSON response and extract token content
                        json_data = json.loads(output[6:])["choices"][0]
                        content = json_data["delta"].get("content")

                        if content:
                            token_count += 1

                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                                print(f"{url} First token return time: {first_token_time} seconds")
                                first_token_times.append(first_token_time)

                            if token_count == 30:
                                token_30_time = time.time() - start_time
                                print(f"{url} 30 tokens return time: {token_30_time} seconds")
                                token_30_times.append(token_30_time)
                            elif token_count == 300:
                                token_300_time = time.time() - start_time
                                print(f"{url} 300 tokens return time: {token_300_time} seconds")
                                token_300_times.append(token_300_time)
                            elif token_count == 3000:
                                token_3000_time = time.time() - start_time
                                print(f"{url} 3000 tokens return time: {token_3000_time} seconds")
                                token_3000_times.append(token_3000_time)

                        # Check if the response has finished
                        finish_reason = json_data["finish_reason"]
                        if finish_reason == "stop":
                            last_token_time = time.time() - start_time
                            print(f"{url} The final token return time: {last_token_time} seconds")
                            final_token_times.append(last_token_time)  # Save the final token time
                            break
                    except:
                        pass

                # Padding
                while len(token_30_times) < len(first_token_times):
                    token_30_times.append("")

                while len(token_300_times) < len(first_token_times):
                    token_300_times.append("")

                while len(token_3000_times) < len(first_token_times):
                    token_3000_times.append("")

                while len(final_token_times) < len(first_token_times):
                    last_token_time = time.time() - start_time
                    final_token_times.append(last_token_time)  # Save the final token time

            else:
                response.failure(f"Failed with status code: {response.status_code}")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # Simulate waiting time between user actions
    host = "http://192.168.8.119"


# This event listener triggers when the test stops
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print(first_token_times)
    # Save the first_token_times and final_token_times to a CSV file
    with open("1gpu_1p_token_times.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["First Token Time", "30 Tokens Time", "300 Tokens Time", "3000 Tokens Time", "Final Token Time"])

        for first, token_30, token_300, token_3000, final in zip(first_token_times, token_30_times, token_300_times, token_3000_times, final_token_times):
            writer.writerow([first, token_30, token_300, token_3000, final])

