"""
Advanced AI Agent Application for Dr. Meenakshi Tomar Dental Chatbot
Integrates all advanced features: Multi-Agent System, Chain-of-Thought, Reprompting, Memory Management
"""

import os
import uuid
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from loguru import logger

# AgentOps compatibility
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    logger.warning("AgentOps not available. Monitoring features disabled.")
    AGENTOPS_AVAILABLE = False

    class DummyAgentOps:
        @staticmethod
        def init(*args, **kwargs):
            pass

        @staticmethod
        def end_session(*args, **kwargs):
            pass

    agentops = DummyAgentOps()

# Import our advanced modules
from multi_agent_system import MultiAgentOrchestrator, AgentResponse
from memory_manager import MemoryManager, MessageType
from advanced_prompts import QueryClassifier, QueryType
from quality_checker import ResponseQualityChecker
from chat_logger import chat_logger

# Load environment variables
load_dotenv()

# Configure logging
logger.add("logs/advanced_chatbot.log", rotation="1 day", retention="7 days", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Dr. Meenakshi Tomar Advanced Dental Chatbot",
    description="AI-powered dental consultation with multi-agent system and advanced reasoning",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Global variables
orchestrator: Optional[MultiAgentOrchestrator] = None
memory_manager: Optional[MemoryManager] = None
openai_client: Optional[openai.OpenAI] = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionInfo(BaseModel):
    session_id: str
    total_interactions: int
    main_concerns: list
    mentioned_symptoms: list
    last_interaction: str

@app.on_event("startup")
async def startup_event():
    """Initialize all advanced components on startup"""
    global orchestrator, memory_manager, openai_client
    
    try:
        logger.info("🚀 Starting Advanced Dr. Chatbot server...")
        
        # Initialize OpenAI client
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        logger.info("✅ OpenAI client initialized")
        
        # Monitoring via standard logging
        logger.info("✅ Standard logging monitoring initialized")
        
        # Initialize Memory Manager
        memory_manager = MemoryManager(storage_dir="conversation_memory")
        logger.info("✅ Memory Manager initialized")
        
        # Initialize Multi-Agent Orchestrator
        orchestrator = MultiAgentOrchestrator(
            openai_client=openai_client,
            pinecone_api_key=os.getenv('PINECONE_API_KEY')
        )
        logger.info("✅ Multi-Agent Orchestrator initialized")
        
        # Cleanup expired sessions
        memory_manager.cleanup_expired_sessions()
        
        logger.info("🎉 Advanced AI Agent System ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

@app.get("/")
async def read_root():
    """Serve the enhanced modern chat UI"""
    return FileResponse("modern_chat.html")

@app.get("/logs")
async def logs_viewer():
    """Serve the chat logs viewer"""
    return FileResponse("log_viewer.html")

@app.get("/advanced_chat.html")
async def advanced_chat():
    """Serve the advanced chat interface"""
    return FileResponse("advanced_chat.html")

@app.get("/modern_chat.html")
async def modern_chat():
    """Serve the basic chat interface"""
    return FileResponse("modern_chat.html")

@app.post("/chat", response_model=ChatResponse)
async def advanced_chat_endpoint(request: ChatRequest):
    """Advanced chat endpoint with multi-agent processing"""
    
    import time
    start_time = time.time()
    
    try:
        logger.info(f"📝 Received message: {request.message[:100]}...")
        
        # Get or create session
        if request.session_id:
            session_id = request.session_id
        else:
            session_id = memory_manager.get_session_id(request.user_id)
        
        # Get conversation memory
        conversation = memory_manager.get_conversation(session_id)
        
        # Add user message to memory
        conversation.add_message(request.message, MessageType.USER)
        
        # Get conversation context and history
        conversation_context = conversation.get_conversation_summary()
        
        # Get last 4 messages for context (2 user + 2 assistant)
        messages = getattr(conversation, 'messages', [])
        if not isinstance(messages, list):
            messages = []
        recent_messages = messages[-4:] if len(messages) > 0 else []
        conversation_history = "\n".join([
            f"{'User' if msg.message_type.value == 'user' else 'Dr. Tomar'}: {msg.content}"
            for msg in recent_messages
        ])
        
        # Process with multi-agent system
        agent_response: AgentResponse = orchestrator.process_consultation(request.message, conversation_context)
        
        # Add assistant response to memory
        conversation.add_message(
            agent_response.content, 
            MessageType.ASSISTANT,
            metadata={
                "agent_type": agent_response.agent_type.value,
                "confidence": agent_response.confidence,
                "quality_score": agent_response.quality_score,
                "attempts_used": agent_response.attempts_used
            }
        )
        
        # Save conversation
        memory_manager.save_conversation(session_id)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"✅ Response generated - Agent: {agent_response.agent_type.value}, "
                   f"Quality: {agent_response.quality_score:.1f}, "
                   f"Time: {response_time_ms}ms")

        # Log detailed interaction to file
        chat_logger.log_chat_interaction(
            session_id=session_id,
            user_message=request.message,
            ai_response=agent_response.content,
            agent_type=agent_response.agent_type.value,
            confidence=agent_response.confidence,
            quality_score=agent_response.quality_score,
            attempts_used=agent_response.attempts_used,
            response_time_ms=response_time_ms,
            reasoning_steps=agent_response.reasoning_steps,
            context_chunks=len(conversation_context.split('\n')) if conversation_context else 0,
            conversation_context=conversation_context
        )

        # Return simple response (no metadata in UI)
        return ChatResponse(
            response=agent_response.content,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        
        # Fallback response
        return ChatResponse(
            response="I apologize, but I'm experiencing some technical difficulties. Please try again in a moment, or consider calling our office at (425) 775-5162 for immediate assistance.",
            session_id=session_id if 'session_id' in locals() else "error",
            agent_type="error",
            confidence=0.0,
            quality_score=0.0,
            reasoning_steps=[],
            conversation_context="",
            response_time_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
        )

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a conversation session"""
    
    try:
        conversation = memory_manager.get_conversation(session_id)
        
        if not conversation.patient_profile:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(
            session_id=session_id,
            total_interactions=conversation.patient_profile.total_interactions,
            main_concerns=conversation.patient_profile.main_concerns,
            mentioned_symptoms=conversation.patient_profile.mentioned_symptoms,
            last_interaction=conversation.patient_profile.last_interaction.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving session information")

@app.post("/session/new")
async def create_new_session(user_id: Optional[str] = None):
    """Create a new conversation session"""
    
    try:
        session_id = memory_manager.get_session_id(user_id)
        conversation = memory_manager.get_conversation(session_id)
        
        return {"session_id": session_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        raise HTTPException(status_code=500, detail="Error creating new session")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Check if all components are initialized
        if not all([orchestrator, memory_manager, openai_client]):
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Components not initialized"}
            )
        
        # Test OpenAI connection
        test_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        return {
            "status": "healthy",
            "components": {
                "orchestrator": "initialized",
                "memory_manager": "initialized",
                "openai_client": "connected",
                "active_sessions": len(memory_manager.active_sessions)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""

    try:
        stats = {
            "active_sessions": len(memory_manager.active_sessions),
            "total_conversations": len(os.listdir(memory_manager.storage_dir)) if os.path.exists(memory_manager.storage_dir) else 0,
            "system_status": "operational"
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": "Unable to retrieve statistics"}

@app.get("/logs/session/{session_id}")
async def get_session_logs(session_id: str):
    """Get detailed logs for a specific session"""

    try:
        session_stats = chat_logger.get_session_stats(session_id)
        return session_stats

    except Exception as e:
        logger.error(f"Error getting session logs: {e}")
        return {"error": "Unable to retrieve session logs"}

@app.get("/logs/daily/{date}")
async def get_daily_logs(date: str):
    """Get daily summary logs (YYYY-MM-DD format)"""

    try:
        daily_summary = chat_logger.get_daily_summary(date)
        return daily_summary

    except Exception as e:
        logger.error(f"Error getting daily logs: {e}")
        return {"error": "Unable to retrieve daily logs"}

@app.get("/logs/daily")
async def get_today_logs():
    """Get today's summary logs"""

    try:
        daily_summary = chat_logger.get_daily_summary()
        return daily_summary

    except Exception as e:
        logger.error(f"Error getting today's logs: {e}")
        return {"error": "Unable to retrieve today's logs"}

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    
    try:
        if memory_manager:
            # Save all active conversations
            for session_id in memory_manager.active_sessions:
                memory_manager.save_conversation(session_id)
            
            logger.info("💾 All conversations saved")
        
        # Cleanup completed via standard logging
        logger.info("Session ended successfully")
            
        logger.info("👋 Advanced AI Agent System shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "advanced_app:app",
        host="0.0.0.0",  # Bind to all interfaces for public access
        port=8000,
        reload=True,
        log_level="info"
    )
