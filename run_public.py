"""
Run the chatbot with public access via ngrok
"""

import os
import subprocess
import sys
import time
import threading
import requests
from loguru import logger

def run_chatbot_server():
    """Run the FastAPI chatbot server"""
    
    logger.info("🚀 Starting Dr. Meenakshi Tomar Chatbot server...")
    
    try:
        # Run the advanced app
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "advanced_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")

def setup_and_run_ngrok():
    """Setup ngrok and create public tunnel"""
    
    auth_token = "31XaV1VBqTp12AujuDnPYoJra48_5ALormMPAtpXC9YwEJYE2"
    
    try:
        # Configure ngrok
        logger.info("🔧 Configuring ngrok...")
        subprocess.run([
            "ngrok", "config", "add-authtoken", auth_token
        ], check=True, capture_output=True)
        
        logger.info("✅ Ngrok configured successfully")
        
        # Wait for server to start
        time.sleep(5)
        
        # Start ngrok tunnel
        logger.info("🌐 Creating public tunnel...")
        process = subprocess.Popen([
            "ngrok", "http", "8000"
        ])
        
        # Wait for ngrok to start
        time.sleep(3)
        
        # Get public URL
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()
            
            if tunnels.get("tunnels"):
                public_url = tunnels["tunnels"][0]["public_url"]
                logger.info(f"🎉 SUCCESS! Your chatbot is now publicly accessible!")
                logger.info(f"🔗 Public URL: {public_url}")
                logger.info(f"📱 Share this link with anyone: {public_url}")
                logger.info("🔄 Keep this running to maintain public access")
                
                # Keep ngrok running
                process.wait()
            else:
                logger.error("❌ No tunnels created")
                
        except Exception as e:
            logger.error(f"❌ Failed to get tunnel info: {e}")
            
    except FileNotFoundError:
        logger.error("❌ Ngrok not found!")
        logger.info("📥 Please install ngrok from: https://ngrok.com/download")
    except Exception as e:
        logger.error(f"❌ Ngrok setup failed: {e}")

def main():
    """Main function to run everything"""
    
    logger.info("🚀 Starting Dr. Meenakshi Tomar Chatbot with Public Access...")
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_chatbot_server, daemon=True)
    server_thread.start()
    
    # Setup ngrok
    setup_and_run_ngrok()

if __name__ == "__main__":
    main()