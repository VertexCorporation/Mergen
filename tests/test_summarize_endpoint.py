"""
Test the summarization endpoint of the monitor server.
"""
import json
import urllib.request
import urllib.error

def test_summarize_endpoint():
    url = "http://localhost:8080/summarize"
    data = {
        "content": "Mergen yapay zeka projesi, sürekli öğrenme ve kod geliştirme yeteneğine sahip bir sistemdir. Bu sistem, 1. sınıftan 12. sınıfa kadar olan seviyelerde çalışarak kendini geliştirir. Her seviyede yeni beceriler öğrenir ve önceki bilgileri birleştirir."
    }
    data_json = json.dumps(data).encode('utf-8')
    
    req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print("Success!")
            print(f"Summary: {result['summary']}")
            print(f"Original length: {result['original_length']}")
            print(f"Summary length: {result['summary_length']}")
            return True
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read().decode('utf-8'))
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_summarize_endpoint()