"""
Multi-Agent System for Specialized Dental Consultation
Implements different specialist agents for various dental domains
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
from loguru import logger
from advanced_prompts import ChainOfThoughtPrompts, QueryClassifier, QueryType
from quality_checker import RepromptingSystem, ResponseQualityChecker
from advanced_rag import AdvancedRAGPipeline
from office_status_helper import check_office_status, get_dynamic_followup_question

# AgentOps compatibility
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    logger.warning("AgentOps not available. Monitoring features disabled.")
    AGENTOPS_AVAILABLE = False

    # Create dummy decorator
    class DummyAgentOps:
        @staticmethod
        def record_function(name):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def init(*args, **kwargs):
            pass

        @staticmethod
        def record_action(*args, **kwargs):
            pass

    agentops = DummyAgentOps()

class AgentType(Enum):
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    EMERGENCY = "emergency"
    SCHEDULING = "scheduling"
    GENERAL = "general"

@dataclass
class AgentResponse:
    content: str
    confidence: float
    agent_type: AgentType
    reasoning_steps: List[str]
    quality_score: float
    attempts_used: int

class BaseAgent:
    """Base class for all specialist agents"""
    
    def __init__(self, openai_client, agent_type: AgentType):
        self.client = openai_client
        self.agent_type = agent_type
        self.cot_prompts = ChainOfThoughtPrompts()
        self.reprompting_system = RepromptingSystem(openai_client)
        
    def get_specialist_persona(self) -> str:
        """Get specialized persona for this agent type"""
        base_persona = self.cot_prompts._get_base_persona()
        
        specializations = {
            AgentType.DIAGNOSTIC: """
DIAGNOSTIC SPECIALIZATION:
- Dr. Tomar is expert in symptom analysis and differential diagnosis
- Dr. Tomar is skilled in identifying urgent vs non-urgent conditions
- Dr. Tomar is experienced in pain assessment and oral pathology
- Dr. Tomar focuses on thorough symptom evaluation and risk assessment
""",
            AgentType.TREATMENT: """
TREATMENT SPECIALIZATION:
- Dr. Tomar is expert in comprehensive treatment planning
- Dr. Tomar is skilled in explaining complex procedures clearly
- Dr. Tomar is experienced in treatment options and alternatives
- Dr. Tomar focuses on patient education and informed consent
""",
            AgentType.PREVENTION: """
PREVENTION SPECIALIZATION:
- Dr. Tomar is expert in preventive dentistry and oral hygiene
- Dr. Tomar is skilled in patient education and behavior modification
- Dr. Tomar is experienced in risk factor assessment and management
- Dr. Tomar focuses on long-term oral health maintenance
""",
            AgentType.EMERGENCY: """
EMERGENCY SPECIALIZATION:
- Dr. Tomar is expert in dental emergency assessment and triage
- Dr. Tomar is skilled in pain management and urgent care protocols
- Dr. Tomar is experienced in trauma and acute condition management
- Dr. Tomar focuses on immediate care and stabilization
""",
            AgentType.SCHEDULING: """
SCHEDULING SPECIALIZATION:
- Dr. Tomar's office scheduling expert with real-time availability
- Expert in appointment booking, office hours, and availability checking
- Skilled in timezone calculations for Edmonds, Washington (America/Los_Angeles)
- Experienced in handling appointment requests, cancellations, and rescheduling
- Focuses on accurate day/time calculations and office status updates

OFFICE HOURS & SCHEDULE:
• Monday: 7:00 AM - 6:00 PM (OPEN)
• Tuesday: 7:00 AM - 6:00 PM (OPEN) 
• Wednesday: CLOSED
• Thursday: 7:00 AM - 6:00 PM (OPEN)
• Friday: CLOSED
• Saturday: CLOSED
• Sunday: CLOSED

LOCATION: Edmonds, Washington (Pacific Time Zone - America/Los_Angeles)
PHONE: (425) 775-5162
""",
            AgentType.GENERAL: """
GENERAL CONSULTATION SPECIALIZATION:
- Dr. Tomar is expert in comprehensive dental care coordination
- Dr. Tomar is skilled in patient communication and education
- Dr. Tomar is experienced in holistic oral health assessment
- Dr. Tomar focuses on overall patient wellbeing and care continuity.
- Dr tomar only give Dental related problem solution not out of the context.
"""
        }
        
        return base_persona + specializations.get(self.agent_type, "")
    
    def process_query(
        self, 
        user_question: str, 
        context: str = "",
        query_type: QueryType = None,
        conversation_history: str = ""
    ) -> AgentResponse:
        """Process query with specialized approach"""
        
        # Map QueryType to AgentType if not already specified
        if query_type:
            agent_query_type = query_type
        else:
            classifier = QueryClassifier(self.client)
            agent_query_type = classifier.classify_query(user_question, self.client)
        
        # Generate specialized prompt with conversation history
        specialist_persona = self.get_specialist_persona()
        cot_prompt = self.cot_prompts.get_chain_of_thought_prompt(
            agent_query_type, user_question, context, conversation_history
        )
        
        # Combine specialist persona with chain-of-thought prompt
        full_prompt = f"{specialist_persona}\n\n{cot_prompt}"
        
        # Generate initial response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            initial_response = response.choices[0].message.content
            
            # Clean initial response first
            initial_response = self._remove_meta_commentary(initial_response)
            
            # Improve response through reprompting
            final_response, attempts, quality_scores = self.reprompting_system.improve_response_with_reprompting(
                full_prompt, user_question, initial_response, context
            )
            
            # Extract reasoning steps (simplified)
            reasoning_steps = self._extract_reasoning_steps(final_response)
            
            # Calculate confidence based on quality scores
            confidence = self._calculate_confidence(quality_scores)
            
            return AgentResponse(
                content=final_response,
                confidence=confidence,
                agent_type=self.agent_type,
                reasoning_steps=reasoning_steps,
                quality_score=quality_scores[-1].overall_score if quality_scores and len(quality_scores) > 0 else 0,
                attempts_used=attempts
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_type.value} agent: {e}")
            return AgentResponse(
                content="I apologize, but I'm having difficulty processing your question right now. Please try again or consider scheduling an in-person consultation.",
                confidence=0.0,
                agent_type=self.agent_type,
                reasoning_steps=[],
                quality_score=0.0,
                attempts_used=1
            )
    
    def _remove_meta_commentary(self, response: str) -> str:
        """Remove meta-commentary from response"""
        import re
        
        # Patterns to remove
        meta_patterns = [
            r"^.*?this response has been adjusted.*?\n",
            r"^.*?here's a revised response.*?\n",
            r"^.*?certainly! here.*?\n",
            r"^.*?response that incorporates.*?\n",
            r"^.*?adjusted to reflect.*?\n",
            r"^.*?more engaging tone.*?\n",
            r"^.*?while maintaining professionalism.*?\n",
            r"^.*?while ensuring clarity.*?\n",
            r"^.*?empathy.*?warmth.*?engaging.*?\n",
            r"^.*?incorporates empathy.*?\n"
        ]
        
        cleaned_response = response
        
        for pattern in meta_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove empty lines at the beginning
        cleaned_response = cleaned_response.lstrip("\n\r ")
        
        return cleaned_response
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        # Simple extraction based on numbered points or bullet points
        lines = response.split('\n')
        reasoning_steps = []
        
        for line in lines:
            line = line.strip()
            if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                line.startswith(('•', '-', '*')) or
                'step' in line.lower()):
                reasoning_steps.append(line)
        
        return reasoning_steps[:5] if reasoning_steps else []  # Limit to 5 steps
    
    def _calculate_confidence(self, quality_scores) -> float:
        """Calculate confidence based on quality scores and attempts"""
        if not quality_scores or len(quality_scores) == 0:
            return 0.5
        
        final_score = quality_scores[-1].overall_score
        attempts = len(quality_scores)
        
        # Higher quality score = higher confidence
        # Fewer attempts needed = higher confidence
        base_confidence = final_score / 100.0
        attempt_penalty = (attempts - 1) * 0.1
        
        confidence = max(0.1, min(1.0, base_confidence - attempt_penalty))
        return confidence

class SchedulingAgent(BaseAgent):
    """Simple scheduling agent with fast responses"""
    
    def __init__(self, openai_client):
        super().__init__(openai_client, AgentType.SCHEDULING)
        
    def get_current_time_info(self) -> Dict[str, any]:
        """Get current time info"""
        from datetime import datetime, timedelta
        import pytz
        
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        tomorrow = now + timedelta(days=1)
        
        return {
            'current_day': now.strftime('%A'),
            'tomorrow_day': tomorrow.strftime('%A'),
            'hour': now.hour,
            'time_str': now.strftime('%I:%M %p')
        }
    
    def get_next_open_day(self, current_day: str) -> str:
        """Get next open day"""
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        open_days = {'Monday', 'Tuesday', 'Thursday'}
        
        idx = days.index(current_day)
        for i in range(1, 8):
            next_day = days[(idx + i) % 7]
            if next_day in open_days:
                return next_day
        return 'Tuesday'
    
    def detect_scheduling_intent(self, user_question: str) -> str:
        """Fast intent detection"""
        q = user_question.lower()
        
        if any(word in q for word in ['today', 'same day', 'aaj']):
            return 'same_day_request'
        elif any(word in q for word in ['tomorrow', 'next']):
            return 'tomorrow_request'
        elif any(phrase in q for phrase in ['can you see me', 'see me']):
            return 'see_me_request'
        elif any(word in q for word in ['hours', 'open', 'close']):
            return 'hours_inquiry'
        elif any(word in q for word in ['cancel', 'reschedule']):
            return 'modify_appointment'
        elif any(word in q for word in ['cost', 'price', 'fee']):
            return 'cost_inquiry'
        elif any(word in q for word in ['insurance', 'coverage']):
            return 'insurance_inquiry'
        else:
            return 'schedule_request'
    
    def generate_response(self, intent: str, time_info: Dict) -> str:
        """Generate responses with proper office hours logic"""
        current_day = time_info['current_day']
        tomorrow_day = time_info.get('tomorrow_day', '')
        hour = time_info['hour']
        next_open = self.get_next_open_day(current_day)
        
        # Check office status
        is_open_day = current_day in ['Monday', 'Tuesday', 'Thursday']
        is_office_hours = 7 <= hour < 18
        is_open = is_open_day and is_office_hours
        
        # Check tomorrow status
        is_tomorrow_open = tomorrow_day in ['Monday', 'Tuesday', 'Thursday']
        
        if intent == 'same_day_request':
            if not is_open_day:
                return f"Same-day appointments not available today ({current_day} - office closed). Next available: {next_open} 7 AM-6 PM. Call: (425) 775-5162"
            elif is_open:
                return "Call now for same-day availability: (425) 775-5162. Currently open until 6 PM today."
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM for same-day appointments. Call: (425) 775-5162"
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Next available: {next_open} 7 AM-6 PM. Call: (425) 775-5162"
        
        elif intent == 'see_me_request':
            if not is_open_day:
                return f"Dr. Tomar's office is closed today ({current_day}). Next availability: {next_open} 7 AM-6 PM. Call: (425) 775-5162"
            elif is_open:
                return "Dr. Tomar may be available today. Call now: (425) 775-5162. Currently open until 6 PM."
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Call: (425) 775-5162 to check availability."
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Next availability: {next_open} 7 AM-6 PM. Call: (425) 775-5162"
        
        elif intent == 'hours_inquiry':
            if not is_open_day:
                return f"Office closed today ({current_day}). Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Closed: Wed/Fri/Weekend. Call: (425) 775-5162"
            elif is_open:
                return "Currently open until 6 PM today! Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Closed: Wed/Fri/Weekend. Call: (425) 775-5162"
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Call: (425) 775-5162"
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Next open: {next_open} 7 AM-6 PM. Call: (425) 775-5162"
        
        elif intent == 'modify_appointment':
            if not is_open_day:
                return f"Office closed today ({current_day}). To cancel/reschedule, call: (425) 775-5162. Next open: {next_open} 7 AM-6 PM"
            elif is_open:
                return "Call now to cancel/reschedule: (425) 775-5162. Currently open until 6 PM today."
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Call: (425) 775-5162 to cancel/reschedule."
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). To cancel/reschedule, call: (425) 775-5162. Next open: {next_open} 7 AM-6 PM"
        
        elif intent == 'cost_inquiry':
            return "For pricing information, please call: (425) 775-5162. We'll discuss costs during consultation."
        
        elif intent == 'insurance_inquiry':
            return "For insurance coverage details, call: (425) 775-5162. We accept most major insurance plans."
        
        elif intent == 'tomorrow_request':
            if is_tomorrow_open:
                return f"Yes, we are open tomorrow ({tomorrow_day}) 7 AM-6 PM. Call (425) 775-5162 to schedule your appointment."
            else:
                next_after_tomorrow = self.get_next_open_day(tomorrow_day)
                return f"We are closed tomorrow ({tomorrow_day}). Next available: {next_after_tomorrow} 7 AM-6 PM. Call: (425) 775-5162"
        
        else:  # schedule_request
            if not is_open_day:
                return f"Office closed today ({current_day}). Call (425) 775-5162 to schedule. Next available: {next_open} 7 AM-6 PM."
            elif is_open:
                return "Call now to schedule: (425) 775-5162. Currently open until 6 PM today."
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Call: (425) 775-5162 to schedule."
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Call (425) 775-5162 to schedule. Next available: {next_open} 7 AM-6 PM."
    
    def process_scheduling_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Process scheduling queries"""
        
        # Check for location questions
        if 'location' in user_question.lower():
            return AgentResponse(
                content="Yes, Dr. Tomar has a second location:\n\n**Pacific Highway Dental**\n\n27020 Pacific Highway South, Suite C\nKent, WA 98032\nPhone: (253) 529-9434",
                confidence=1.0,
                agent_type=AgentType.SCHEDULING,
                reasoning_steps=["Location question detected"],
                quality_score=100.0,
                attempts_used=1
            )
        
        # Get time info and detect intent
        time_info = self.get_current_time_info()
        intent = self.detect_scheduling_intent(user_question)
        content = self.generate_response(intent, time_info)
        
        return AgentResponse(
            content=content,
            confidence=0.95,
            agent_type=AgentType.SCHEDULING,
            reasoning_steps=[f"Intent: {intent}", f"Day: {time_info['current_day']}"],
            quality_score=95.0,
            attempts_used=1
        )

class MultiAgentOrchestrator:
    """Orchestrates multiple specialist agents for optimal responses"""
    
    def __init__(self, openai_client, pinecone_api_key: str):
        self.client = openai_client
        self.rag_pipeline = AdvancedRAGPipeline(openai_client, pinecone_api_key)
        self.query_classifier = QueryClassifier(openai_client)
        
        # Initialize specialist agents
        self.agents = {
            AgentType.DIAGNOSTIC.value: BaseAgent(openai_client, AgentType.DIAGNOSTIC),
            AgentType.TREATMENT.value: BaseAgent(openai_client, AgentType.TREATMENT),
            AgentType.PREVENTION.value: BaseAgent(openai_client, AgentType.PREVENTION),
            AgentType.EMERGENCY: BaseAgent(openai_client, AgentType.EMERGENCY),
            AgentType.SCHEDULING: SchedulingAgent(openai_client),
            AgentType.GENERAL: BaseAgent(openai_client, AgentType.GENERAL)
        }
        
        logger.info("Multi-Agent System initialized with all specialist agents")
    
    def route_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Route query to appropriate specialist agent"""
        
        # AI-powered query classification
        query_type = self.query_classifier.classify_query(user_question, self.client)
        
        # Route to scheduling agent if classified as scheduling
        if query_type == QueryType.SCHEDULING:
            scheduling_agent = self.agents[AgentType.SCHEDULING]
            return scheduling_agent.process_scheduling_query(user_question, context)
        
        # Route to appropriate agent
        agent = self.agents.get(query_type.value, self.agents[AgentType.GENERAL])
        
        # Get relevant context from RAG if needed
        if context:
            rag_context = context
        else:
            rag_context = self.rag_pipeline.get_relevant_context(user_question)
        
        return agent.process_query(user_question, rag_context, query_type)
    
    def process_consultation(self, user_question: str, conversation_history: str = "") -> AgentResponse:
        """Main consultation processing with agent orchestration"""
        
        logger.info(f"Processing consultation: {user_question[:50]}...")
        
        # Check for greeting messages
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if user_question.lower().strip() in greeting_words:
            return AgentResponse(
                content="Welcome to Edmonds Bay Dental! How can I help you today?",
                confidence=1.0,
                agent_type=AgentType.GENERAL,
                reasoning_steps=["Detected greeting message"],
                quality_score=100.0,
                attempts_used=1
            )
        
        # 1. Retrieve relevant context
        context, retrieved_chunks = self.rag_pipeline.retrieve_and_rank(user_question)
        
        # 2. Route to appropriate agent and get response directly
        response = self.route_query(user_question, context)
        
        logger.info(f"Routed to {response.agent_type.value} agent")
        
        return response

# Example usage
if __name__ == "__main__":
    # This would be integrated into the main application
    pass