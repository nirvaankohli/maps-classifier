import requests

API_KEY = "f9c03b2f-9c57-4396-a94a-25abc1d9c1d1"
URL = "https://hackatime.hackclub.com/api/v1/stats"

headers = {
    f"Authorization": f"Bearer {API_KEY}"
}

response = requests.get(URL, headers=headers)

if response.ok:
    stats = response.json()
    print("Stats:", stats)
else:
    print(f"Request failed: {response.status_code}\n{response.text}")
