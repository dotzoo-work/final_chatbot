@echo off
echo 🚀 Starting Dr. Meenakshi Tomar Chatbot with Public Access...
echo.

REM Configure ngrok auth token
echo 🔧 Configuring ngrok...
ngrok config add-authtoken 31XaV1VBqTp12AujuDnPYoJra48_5ALormMPAtpXC9YwEJYE2

REM Start the chatbot server in background
echo 🚀 Starting chatbot server...
start /B python -m uvicorn advanced_app:app --host 0.0.0.0 --port 8000 --reload

REM Wait for server to start
echo ⏳ Waiting for server to start...
timeout /t 5 /nobreak >nul

REM Start ngrok tunnel
echo 🌐 Creating public tunnel...
ngrok http 8000

pause