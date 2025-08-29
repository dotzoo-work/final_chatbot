"""
Setup script to configure ngrok and make the chatbot publicly accessible
"""

import os
import subprocess
import sys
import time
import requests
from loguru import logger

def setup_ngrok():
    """Configure ngrok with the provided auth token"""
    
    auth_token = "31XaV1VBqTp12AujuDnPYoJra48_5ALormMPAtpXC9YwEJYE2"
    
    try:
        # Configure ngrok auth token
        logger.info("🔧 Configuring ngrok auth token...")
        result = subprocess.run([
            "ngrok", "config", "add-authtoken", auth_token
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Ngrok auth token configured successfully")
        else:
            logger.error(f"❌ Failed to configure ngrok: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ Ngrok not found. Please install ngrok first:")
        logger.info("Download from: https://ngrok.com/download")
        return False
    
    return True

def start_ngrok_tunnel(port=8000):
    """Start ngrok tunnel for the specified port"""
    
    try:
        logger.info(f"🚀 Starting ngrok tunnel on port {port}...")
        
        # Start ngrok tunnel
        process = subprocess.Popen([
            "ngrok", "http", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for ngrok to start
        time.sleep(3)
        
        # Get the public URL
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()
            
            if tunnels.get("tunnels"):
                public_url = tunnels["tunnels"][0]["public_url"]
                logger.info(f"🌐 Public URL: {public_url}")
                logger.info(f"🔗 Your chatbot is now accessible at: {public_url}")
                return public_url, process
            else:
                logger.error("❌ No tunnels found")
                return None, process
                
        except Exception as e:
            logger.error(f"❌ Failed to get tunnel info: {e}")
            return None, process
            
    except Exception as e:
        logger.error(f"❌ Failed to start ngrok: {e}")
        return None, None

def main():
    """Main setup function"""
    
    logger.info("🚀 Setting up public access for Dr. Meenakshi Tomar Chatbot...")
    
    # Setup ngrok
    if not setup_ngrok():
        sys.exit(1)
    
    # Start tunnel
    public_url, process = start_ngrok_tunnel()
    
    if public_url:
        logger.info("✅ Setup complete!")
        logger.info(f"📱 Share this URL: {public_url}")
        logger.info("🔄 Keep this script running to maintain the tunnel")
        
        try:
            # Keep the script running
            while True:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping ngrok tunnel...")
            if process:
                process.terminate()
            logger.info("👋 Tunnel stopped")
    else:
        logger.error("❌ Failed to create public tunnel")
        sys.exit(1)

if __name__ == "__main__":
    main()