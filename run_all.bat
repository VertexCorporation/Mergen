@echo off
chcp 65001 >nul
echo ============================================
echo  MERGEN V2 - HEPSINI BAŞLAT
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
pip install requests -q >nul 2>&1

echo.
echo ============================================
echo  Sistem baslatiliyor...
echo  API Anahtarlari bellekte tutuluyor
echo ============================================
echo.

set OPENROUTER_KEY=sk-or-v1-dummy
set TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567

echo [1/3] Otomatik Gelisim baslatiliyor...
start "Mergen Evolution" cmd /k "title Mergen Evolution && cd /d %~dp0 && python auto_evolution.py %OPENROUTER_KEY% %TELEGRAM_TOKEN%"

timeout /t 5 >nul

echo [2/3] Telegram Bot baslatiliyor...
start "Telegram Bot" cmd /k "title Telegram Bot && cd /d %~dp0 && python telegram_bot.py %TELEGRAM_TOKEN%"

timeout /t 3 >nul

echo [3/3] Monitor Sunucusu baslatiliyor...
start "Monitor Server" cmd /k "title Monitor Server && cd /d %~dp0 && python monitor_server.py"

echo.
echo ============================================
echo  TUM SISTEMLER BASLATILDI!
echo ============================================
echo.
echo  * Mergen Gelisimi: Yeni acilan pencereye bakin
echo  * Telegram: Botunuza "rapor" yazin
echo  * Monitor: tarayicinizda http://localhost:8080 acin
echo.
echo Kapatmak icin bu pencereyi kapatin, acilan pencereleri ayri ayri kapatmaniz gerekir.
echo.
pause
