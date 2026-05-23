import urllib.request
import json

def list_models():
    key = "AIzaSyBjOMSP6Bl9jZKxgJVZh6X1Ca38wkjiD48"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    
    print("Listing models...")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            for m in res_data.get("models", []):
                print("-", m.get("name"), m.get("supportedGenerationMethods"))
    except Exception as e:
        print("Error in v1beta listing:")
        import traceback
        traceback.print_exc()
        if hasattr(e, "read"):
            try:
                print("Body:", e.read().decode("utf-8"))
            except:
                pass
                
    url_v1 = f"https://generativelanguage.googleapis.com/v1/models?key={key}"
    print("\nListing models for v1...")
    try:
        req = urllib.request.Request(url_v1, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            for m in res_data.get("models", []):
                print("-", m.get("name"), m.get("supportedGenerationMethods"))
    except Exception as e:
        print("Error in v1 listing:")
        if hasattr(e, "read"):
            try:
                print("Body:", e.read().decode("utf-8"))
            except:
                pass

if __name__ == "__main__":
    list_models()
