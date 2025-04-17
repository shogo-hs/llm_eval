import requests
import json

def test_llm_connection():
    """ローカルLLMとの疎通テスト"""
    url = "http://192.168.3.43:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "elyza",
        "messages": [{"role": "user", "content": "日本の四季について短く説明してください"}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_llm_connection()
