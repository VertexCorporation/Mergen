"""
Basit Monitor Sunucusu - index.html ve status.json servis eder
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class MonitorHandler(SimpleHTTPRequestHandler):
    """Basit HTTP handler - statik dosyaları servis eder."""
    
    def do_GET(self):
        """GET isteklerini işle."""
        if self.path == '/' or self.path == '/index.html':
            # index.html serve et
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except FileNotFoundError:
                self.send_error(404, 'index.html bulunamadı')
        
        elif self.path.startswith('/status.json'):
            # status.json servis et
            try:
                status_path = Path('status.json')
                if status_path.exists():
                    with open(status_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
                else:
                    # Varsayılan durum
                    default_status = {
                        "current_grade": 1,
                        "stats": {
                            "total_improvements": 0,
                            "total_errors": 0,
                            "models_used": [],
                            "files_modified": []
                        },
                        "evolution_log": []
                    }
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(default_status, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f'Hata: {e}')
        
        else:
            # Diğer dosyaları varsayılan olarak servis et
            super().do_GET()
    
    def log_message(self, format, *args):
        """Log mesajlarını özelleştir."""
        print(f"[Monitor] {args[0]} {args[1]} {args[2]}")


if __name__ == '__main__':
    print("=" * 60)
    print("🧠 MERGEN V2 - Monitor Sunucusu")
    print("=" * 60)
    print("Sunucu başlatılıyor... http://localhost:8080")
    print("İzleme için tarayıcıda açın: http://localhost:8080")
    print("Çıkmak için Ctrl+C")
    print("=" * 60)
    
    server = HTTPServer(('0.0.0.0', 8080), MonitorHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatılıyor...")
        server.shutdown()
        print("Sunucu kapatıldı.")
