@echo off
echo ============================================
echo  MERGEN V2 - OTOMATIK GELISIM SISTEMI
echo ============================================
echo.

REM Python kontrolü
python --version >nul 2>&1
if errorlevel 1 (
    echo Python bulunamadi! Lutfen Python yukleyin.
    pause
    exit /b 1
)

REM Gerekli paketleri yukle
echo Gerekli paketler kontrol ediliyor...
pip install requests -q

echo.
echo ============================================
echo  Sistem baslatiliyor...
echo  API Anahtarlari bellekte tutuluyor (dosyaya YAZILMAZ)
echo ============================================
echo.

REM OpenRouter API Key
set OPENROUTER_KEY=sk-or-v1-dummy

REM Telegram Bot Token
set TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567

echo 1. Otomatik Gelisim (auto_evolution.py)
echo 2. Telegram Bot (telegram_bot.py)
echo 3. Monitor Sunucusu (monitor_server.py)
echo.
set /p SECIM="Seciminiz (1-3, hepsi icin Enter): "

if "%SECIM%"=="" (
    REM Hepsi
    start "Auto Evolution" cmd /k "python auto_evolution.py %OPENROUTER_KEY% %TELEGRAM_TOKEN%"
    timeout /t 3 >nul
    start "Telegram Bot" cmd /k "python telegram_bot.py %TELEGRAM_TOKEN%"
    timeout /t 3 >nul
    start "Monitor" cmd /k "python monitor_server.py"
) else if "%SECIM%"=="1" (
    python auto_evolution.py %OPENROUTER_KEY% %TELEGRAM_TOKEN%
) else if "%SECIM%"=="2" (
    python telegram_bot.py %TELEGRAM_TOKEN%
) else if "%SECIM%"=="3" (
    python monitor_server.py
)

pause
