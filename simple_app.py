from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from advanced_prompts import ChainOfThoughtPrompts
from datetime import datetime
import pytz

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    current_time: str = None
    day_of_week: str = None
    tomorrow_day: str = None
    timezone: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str = None

# Initialize OpenAI client in function to avoid startup errors
def get_openai_client():
    return openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.get("/")
async def read_root():
    return FileResponse("modern_chat.html")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"Received message: {request.message}")
        client = get_openai_client()
        print("OpenAI client created")
        
        # Get current California time on server side
        from datetime import timedelta
        california_tz = pytz.timezone('America/Los_Angeles')
        california_now = datetime.now(california_tz)
        today_ca = california_now.strftime('%A')
        tomorrow_ca = (california_now + timedelta(days=1)).strftime('%A')
        
        # Add time context to the message
        message_with_context = request.message
        time_context = f"\n\n🚨 CRITICAL TIME CONTEXT - YOU MUST USE THIS EXACT INFORMATION:\n\n📅 TODAY IS: {today_ca}\n📅 TOMORROW IS: {tomorrow_ca}\n🕐 Current California time: {california_now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n🏥 CLINIC SCHEDULE (EXACT DAYS):\n• Monday: 7AM-6PM (OPEN)\n• Tuesday: 7AM-6PM (OPEN)\n• Wednesday: CLOSED\n• Thursday: 7AM-6PM (OPEN)\n• Friday: CLOSED\n• Saturday: CLOSED\n• Sunday: CLOSED\n\n⚠️ SCHEDULING RULE: If user asks about 'tomorrow', check what day tomorrow is ({tomorrow_ca}) and respond accordingly:\n- If {tomorrow_ca} is OPEN: Say 'Tomorrow is {tomorrow_ca} - Open (7AM-6PM)'\n- If {tomorrow_ca} is CLOSED: Say 'Tomorrow is {tomorrow_ca} - Closed'\n\n🔴 DO NOT use any other day information - ONLY use the days specified above!"
        message_with_context += time_context
        
        print(f"\n=== CALIFORNIA TIME CONTEXT DEBUG ===")
        print(f"California Now: {california_now}")
        print(f"Today: {today_ca}")
        print(f"Tomorrow: {tomorrow_ca}")
        print(f"Tomorrow Status: {'OPEN' if tomorrow_ca in ['Monday', 'Tuesday', 'Thursday'] else 'CLOSED'}")
        print(f"=== END DEBUG ===")
        
        # Use advanced prompts
        advanced_prompts = ChainOfThoughtPrompts()
        messages = [
            {
                "role": "system",
                "content": advanced_prompts.base_persona
            },
            {
                "role": "user", 
                "content": message_with_context
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        
        print("OpenAI response received")
        print(f"California time context was included in message to AI")
        session_id = request.session_id or "default_session"
        return ChatResponse(response=response.choices[0].message.content, session_id=session_id)
    except Exception as e:
        print(f"Error: {str(e)}")
        return ChatResponse(response=f"Sorry, I encountered an error: {str(e)}", session_id=request.session_id or "default_session")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)