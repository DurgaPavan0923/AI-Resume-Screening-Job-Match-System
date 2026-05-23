import urllib.request
import json
import os

def test_gemini():
    key = "AIzaSyBjOMSP6Bl9jZKxgJVZh6X1Ca38wkjiD48"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Hello, respond with exactly 'Success'"}
                ]
            }
        ]
    }
    
    print("Sending request to Gemini API...")
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print("Response:", res_data)
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            print("Extracted text:", text)
    except Exception as e:
        print("Error occurred:")
        import traceback
        traceback.print_exc()
        if hasattr(e, "read"):
            try:
                print("Error body:", e.read().decode("utf-8"))
            except:
                pass

if __name__ == "__main__":
    test_gemini()
