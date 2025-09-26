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

class GeneralAgent(BaseAgent):
    """General consultation agent with real-time availability"""
    
    def __init__(self, openai_client):
        super().__init__(openai_client, AgentType.GENERAL)
        
    def get_current_time_info(self) -> Dict[str, str]:
        """Get current time information for Edmonds, Washington"""
        from datetime import datetime, timedelta
        import pytz
        
        # Edmonds, Washington timezone
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        
        return {
            'current_time': now.strftime('%I:%M %p'),
            'current_date': now.strftime('%A, %B %d, %Y'),
            'current_day': now.strftime('%A'),
            'timezone': 'Pacific Time (America/Los_Angeles)'
        }
    
    def get_next_open_day(self) -> str:
        """Get the next open day from today"""
        from datetime import datetime, timedelta
        import pytz
        
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        open_days = {'Monday', 'Tuesday', 'Thursday'}
        
        # Check next 7 days
        for i in range(1, 8):
            next_date = now + timedelta(days=i)
            next_day = next_date.strftime('%A')
            if next_day in open_days:
                return next_day
        
        return 'Monday'  # fallback
    
    def process_general_query(self, user_question: str, context: str = "", conversation_history: str = "") -> AgentResponse:
        """Process general queries with real-time availability context"""
        
        # Get current time info
        time_info = self.get_current_time_info()
        current_day_status = check_office_status(time_info['current_day'])
        
        # Add real-time context to the prompt
        specialist_persona = self.get_specialist_persona()
        
        # Add current office status to context
        realtime_context = f"""
CURRENT OFFICE STATUS:
- Today: {current_day_status['status_message']}
- Current Time: {time_info['current_time']}
- Phone: (425) 775-5162

If mentioning appointments or office availability, use this real-time information.
"""
        
        cot_prompt = self.cot_prompts.get_chain_of_thought_prompt(
            QueryType.GENERAL, user_question, context + realtime_context, conversation_history
        )
        
        # Combine specialist persona with chain-of-thought prompt
        full_prompt = f"{specialist_persona}\n\n{cot_prompt}"
        
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
            logger.error(f"Error in general agent: {e}")
            return AgentResponse(
                content="I apologize, but I'm having difficulty processing your question right now. Please try again or consider calling our office at (425) 775-5162.",
                confidence=0.0,
                agent_type=self.agent_type,
                reasoning_steps=[],
                quality_score=0.0,
                attempts_used=1
            )

class EmergencyAgent(BaseAgent):
    """Specialized agent for dental emergencies with real-time availability"""
    
    def __init__(self, openai_client):
        super().__init__(openai_client, AgentType.EMERGENCY)
        
    def get_current_time_info(self) -> Dict[str, str]:
        """Get current time information for Edmonds, Washington"""
        from datetime import datetime, timedelta
        import pytz
        
        # Edmonds, Washington timezone
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        
        return {
            'current_time': now.strftime('%I:%M %p'),
            'current_date': now.strftime('%A, %B %d, %Y'),
            'current_day': now.strftime('%A'),
            'timezone': 'Pacific Time (America/Los_Angeles)'
        }

    
    def get_tomorrow_day(self) -> str:
        """Get tomorrow's day name"""
        from datetime import datetime, timedelta
        import pytz
        
        pacific_tz = pytz.timezone('America/Los_Angeles')
        tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
        return tomorrow.strftime('%A')
    
    def get_emergency_advice(self, user_question: str) -> str:
        """Get specific emergency advice based on user's problem"""
        question_lower = user_question.lower()
        
        if any(word in question_lower for word in ['broke', 'broken', 'chipped', 'cracked']):
            return "• Rinse mouth with warm water\n• Save any broken pieces\n• Apply cold compress to reduce swelling\n• Avoid chewing on that side"
        elif any(word in question_lower for word in ['pain', 'hurt', 'ache', 'throbbing']):
            return "• Rinse with warm salt water\n• Take over-the-counter pain reliever\n• Apply cold compress for swelling\n• Avoid very hot/cold foods"
        elif any(word in question_lower for word in ['swollen', 'swelling', 'puffy']):
            return "• Apply cold compress for 15-20 minutes\n• Keep head elevated when lying down\n• Rinse with warm salt water\n• Avoid hot foods and drinks"
        elif any(word in question_lower for word in ['bleeding', 'blood']):
            return "• Apply gentle pressure with clean gauze\n• Rinse gently with cold water\n• Apply cold compress to reduce bleeding\n• Avoid spitting or rinsing vigorously"
        elif any(word in question_lower for word in ['knocked out', 'fell out', 'lost tooth']):
            return "• Handle tooth by crown, not root\n• Rinse gently if dirty\n• Try to reinsert or keep in milk\n• Get to dentist within 30 minutes"
        else:
            return None  # No advice if no specific symptoms mentioned
    
    def get_next_open_day(self) -> str:
        """Get the next open day from today"""
        from datetime import datetime, timedelta
        import pytz
        
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        open_days = {'Monday', 'Tuesday', 'Thursday'}
        
        # Check next 7 days
        for i in range(1, 8):
            next_date = now + timedelta(days=i)
            next_day = next_date.strftime('%A')
            if next_day in open_days:
                return next_day
        
        return 'Monday'  # fallback
    
    def process_emergency_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Process emergency queries with real-time availability"""
        
        # Get current time info
        time_info = self.get_current_time_info()
        current_day_status = check_office_status(time_info['current_day'])
        
        # Debug logging
        logger.info(f"🚨 Emergency Request - Day: {time_info['current_day']}, Status: {'OPEN' if current_day_status['is_open'] else 'CLOSED'}")
        logger.info(f"🚨 Emergency Request - Time: {time_info['current_time']}")
        logger.info(f"🚨 Emergency Request - Question: {user_question}")
        
        # Check if user is asking about tomorrow/next or today
        is_future_request = any(word in user_question.lower() for word in ['tomorrow', 'next'])
        is_today_request = any(word in user_question.lower() for word in ['today', 'now', 'right now', 'immediately'])
        
        if is_future_request:
            # Get tomorrow's info
            tomorrow_day = self.get_tomorrow_day()
            tomorrow_day_status = check_office_status(tomorrow_day)
            
            if tomorrow_day_status['is_open']:
                emergency_response = f"""Yes, Dr. Tomar can see you for an emergency appointment tomorrow ({tomorrow_day}) when available.

**Tomorrow's Emergency Availability:**

• Day: {tomorrow_day}
• Hours: 7 AM - 6 PM
• Call: (425) 775-5162 to schedule emergency appointment
• Location: Edmonds Bay Dental, Edmonds, WA

Please call to discuss your emergency and schedule tomorrow's appointment."""
            else:
                next_open = self.get_next_open_day()
                advice = self.get_emergency_advice(user_question)
                if advice:
                    emergency_response = f"""Dr. Tomar's office is closed tomorrow ({tomorrow_day}), but emergency care is important.

**Emergency Options:**

• Call: (425) 775-5162 - Leave emergency message
• Next Open: {next_open} (7 AM - 6 PM)

**For immediate relief:**

{advice}"""
                else:
                    emergency_response = f"""Dr. Tomar's office is closed tomorrow ({tomorrow_day}), but emergency care is important.

**Emergency Options:**

• Call: (425) 775-5162 - Leave emergency message
• Next Open: {next_open} (7 AM - 6 PM)"""
        elif is_today_request or not (is_future_request or is_today_request):
            # Today's emergency request
            if current_day_status['is_open']:
                emergency_response = f"""Dr. Tomar's office is currently open and can schedule emergency appointments when available until 6 PM today.

**Immediate Action:**

• Call now: (425) 775-5162
• Status: {current_day_status['status_message']}
• Location: Edmonds Bay Dental, Edmonds, WA

Please call immediately to discuss your emergency with Dr. Tomar's team."""
            else:
                next_open = self.get_next_open_day()
                advice = self.get_emergency_advice(user_question)
                if advice:
                    emergency_response = f"""Dr. Tomar's office is currently closed, but emergency care is important.

**Current Status:** {current_day_status['status_message']}

**Emergency Options:**

- Call: (425) 775-5162 - Leave an emergency message
- Next Available Appointment: {next_open} (7 AM - 6 PM)

**For immediate relief:**

{advice}"""
                else:
                    emergency_response = f"""Dr. Tomar's office is currently closed, but emergency care is important.

**Current Status:** {current_day_status['status_message']}

**Emergency Options:**

• Call: (425) 775-5162 - Leave emergency message
• Next Available Appointment: {next_open} (7 AM - 6 PM)"""
        
        # Return the template directly without AI processing to avoid unwanted additions
        content = emergency_response
        
        return AgentResponse(
            content=content,
            confidence=0.95,
            agent_type=AgentType.EMERGENCY,
            reasoning_steps=[
                f"Emergency request detected",
                f"Current status: {time_info['current_day']} - {'Open' if current_day_status['is_open'] else 'Closed'}",
                "Generated emergency response with real-time availability"
            ],
            quality_score=95.0,
            attempts_used=1
        )

class SchedulingAgent(BaseAgent):
    """Specialized agent for appointment scheduling and office hours"""
    
    def __init__(self, openai_client):
        super().__init__(openai_client, AgentType.SCHEDULING)
        
    def get_current_time_info(self) -> Dict[str, str]:
        """Get current time information for Edmonds, Washington"""
        from datetime import datetime, timedelta
        import pytz
        
        # Edmonds, Washington timezone
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        
        # Calculate tomorrow
        tomorrow = now + timedelta(days=1)
        
        return {
            'current_time': now.strftime('%I:%M %p'),
            'current_date': now.strftime('%A, %B %d, %Y'),
            'current_day': now.strftime('%A'),
            'tomorrow_day': tomorrow.strftime('%A'),
            'timezone': 'Pacific Time (America/Los_Angeles)'
        }
    
    def get_next_open_day(self, current_day: str) -> str:
        """Get next available open day"""
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        open_days = {'Monday', 'Tuesday', 'Thursday'}
        
        current_index = days.index(current_day)
        
        for i in range(1, 8):
            next_index = (current_index + i) % 7
            next_day = days[next_index]
            if next_day in open_days:
                return next_day
        
        return 'Monday'  # fallback
    
    def is_after_office_hours(self) -> bool:
        """Check if current time is after 6 PM"""
        from datetime import datetime
        import pytz
        
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        return now.hour >= 18  # 6 PM or later
    
    def is_before_office_hours(self) -> bool:
        """Check if current time is before 7 AM"""
        from datetime import datetime
        import pytz
        
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        return now.hour < 7  # Before 7 AM
    
    def generate_closed_day_response(self, tomorrow_day: str, next_open_day: str) -> str:
        """Generate detailed response when office is closed tomorrow"""
        return f"""Dr. Meenakshi Tomar's office is closed tomorrow ({tomorrow_day}). However, the next available appointment day is {next_open_day}.

Please give us a call at (425) 775-5162.

**Scheduling Hours:**

• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Thursday: 7 AM - 6 PM
• Wednesday, Friday, Weekend: Closed

{get_dynamic_followup_question()}"""
    
    def get_direct_response(self, intent: str, time_info: Dict, current_day_status: Dict, tomorrow_day_status: Dict, user_question: str) -> str:
        """Generate direct response without AI interpretation"""
        
        if intent == "see_me_request":
            # Check if today is an open day
            if time_info['current_day'] in ['Monday', 'Tuesday', 'Thursday']:
                if current_day_status['is_open']:
                    return "Yes! Dr. Tomar can see you today. We are open until 6 PM. Please call (425) 775-5162 to schedule your appointment."
                else:
                    # Check time conditions
                    if self.is_after_office_hours():
                        next_day = time_info['tomorrow_day'] if time_info['tomorrow_day'] in ['Monday', 'Tuesday', 'Thursday'] else self.get_next_open_day(time_info['tomorrow_day'])
                        return f"We're currently closed. Next opening is {next_day} at 7 AM.\n\n**Next Available:**\n• Day: {next_day}\n• Hours: 7 AM - 6 PM\n• Phone: (425) 775-5162"
                    elif self.is_before_office_hours():
                        return f"We're currently closed but open today ({time_info['current_day']}) from 7 AM to 6 PM.\n\n**Contact Information:**\n• Phone: (425) 775-5162\n• Location: Edmonds Bay Dental, Edmonds, WA"
            else:
                next_day = self.get_next_open_day(time_info['current_day'])
                return f"Dr. Tomar's office is closed today ({time_info['current_day']}). Our next available day is {next_day}.\n\n**Office Hours:**\n• Monday: 7 AM - 6 PM\n• Tuesday: 7 AM - 6 PM\n• Thursday: 7 AM - 6 PM\n• Wednesday, Friday, Weekend: Closed\n\n**Please Call Us:** (425) 775-5162 for appointments"
        
        elif intent == "same_day_request":
            # Check if today is an open day
            if time_info['current_day'] in ['Monday', 'Tuesday', 'Thursday']:
                if current_day_status['is_open']:
                    return "Yes, we offer same-day appointments! We are currently open until 6 PM. Please call (425) 775-5162 to schedule your same-day appointment."
                else:
                    # Check time conditions
                    if self.is_after_office_hours():
                        next_day = time_info['tomorrow_day'] if time_info['tomorrow_day'] in ['Monday', 'Tuesday', 'Thursday'] else self.get_next_open_day(time_info['tomorrow_day'])
                        return f"Same-day appointments are no longer available today. Next opening is {next_day} at 7 AM.\n\n**Next Available:**\n• Day: {next_day}\n• Phone: (425) 775-5162"
                    elif self.is_before_office_hours():
                        return f"We're currently closed but will open today at 7 AM to 6 PM for same-day appointments.\n\n**Today's Hours:**\n• Opens: 7 AM today\n• Closes: 6 PM today\n• Phone: (425) 775-5162"
            else:
                return f"Same-day appointments are not available today as our office is closed on {time_info['current_day']}s.\n\n**Available Status:**\n• {current_day_status['status_message']}\n• Contact: (425) 775-5162 for scheduling"
        
        return None  # Use AI for other intents
    
    def detect_scheduling_intent(self, user_question: str) -> str:
        """Dynamically detect scheduling intent using AI"""
        
        intent_prompt = f"""
Analyze this user question and classify the scheduling intent. Return ONLY the intent category:

User Question: "{user_question}"

Intent Categories:
- schedule_request: User wants to book/schedule/make an appointment
- same_day_request: User asking about same-day appointments or availability today
- see_me_request: User asking "can you see me" or "can you see me tomorrow" - about being seen by doctor
- hours_inquiry: User asking about office hours, opening times, or when clinic is open
- next_open_request: User asking "when do you open next", "when are you open next", "next time you're open", "when do you open"
- modify_appointment: User wants to cancel, reschedule, or change existing appointment
- cost_inquiry: User asking about prices, costs, or fees for procedures
- insurance_inquiry: User asking about insurance acceptance or coverage
- general_scheduling: Any other scheduling-related question

IMPORTANT DISTINCTIONS:
- "can you see me" = see_me_request (about doctor availability)
- "when do you open" = next_open_request (about office opening times)

Return only the intent category name:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Validate intent
            valid_intents = ["schedule_request", "same_day_request", "see_me_request", 
                           "hours_inquiry", "next_open_request", "modify_appointment", "cost_inquiry", 
                           "insurance_inquiry", "general_scheduling"]
            
            return intent if intent in valid_intents else "general_scheduling"
            
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return "general_scheduling"
    
    def _get_intent_prompt(self, intent: str, time_info: Dict, current_day_status: Dict, tomorrow_day_status: Dict, user_question: str) -> str:
        """Generate response based on specific intent examples"""
        
        context = f"""
CURRENT STATUS:
- Today: {time_info['current_day']} ({'OPEN' if current_day_status['is_open'] else 'CLOSED'})
- Tomorrow: {time_info['tomorrow_day']} ({'OPEN' if tomorrow_day_status['is_open'] else 'CLOSED'})
- Time: {time_info['current_time']}
- Phone: (425) 775-5162
"""
        
        intent_responses = {
            "schedule_request": f"""
{context}
User wants to schedule appointment.

Current Day: {time_info['current_day']}
Tomorrow: {time_info['tomorrow_day']}
Current Time: {time_info['current_time']}
Today Status: {current_day_status['status_message']}
Tomorrow Status: {tomorrow_day_status['status_message']}
Open Days: Monday, Tuesday, Thursday only

IMPORTANT: Detect if user is asking about TODAY or TOMORROW from their question.

CRITICAL LOGIC - Follow this exactly:

If user asks about TOMORROW:
- Check if tomorrow ({time_info['tomorrow_day']}) is an open day (Monday/Tuesday/Thursday)
- If tomorrow is open day: "I’m unable to schedule your appointment directly, but please give us a call and our team can book an appointment when a slot is available"
- If tomorrow is closed day: "Tomorrow ({time_info['tomorrow_day']}) is closed"

If user asks about TODAY or general scheduling:
- Check if today ({time_info['current_day']}) is an open day AND current_day_status['is_open'] is True
- If today is open AND office open: "Our clinic is open right now"
- If today is closed day OR office closed: Show today's status

For TOMORROW requests:
If {time_info['tomorrow_day']} in ['Monday', 'Tuesday', 'Thursday']:
I’m unable to schedule your appointment directly, but please give us a call and our team can book an appointment({time_info['tomorrow_day']}) when a slot is available.

**Tomorrow's Availability:**

• Day: {time_info['tomorrow_day']}
• Open Hours: 7 AM to 6 PM
• Please call: (425) 775-5162 to schedule your appointment

If {time_info['tomorrow_day']} NOT in ['Monday', 'Tuesday', 'Thursday']:
Tomorrow ({time_info['tomorrow_day']}) is closed. Our next available day for scheduling appointments is {self.get_next_open_day(time_info['tomorrow_day'])} from 7 AM to 6 PM. 

 I’m unable to schedule your appointment directly,Our team can schedule your appointment when available. Please call us at (425) 775-5162 to schedule.

**Scheduling Hours:**
• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Thursday: 7 AM - 6 PM
• Wednesday, Friday, Weekend: Closed

Would you like to schedule an appointment for one of our available days? 🦷

For TODAY requests:
If {time_info['current_day']} in ['Monday', 'Tuesday', 'Thursday'] AND current_day_status['is_open'] is True:
Our clinic is open right now! I’m unable to schedule your appointment directly, but please give us a call and our team can book an appointment when a slot is available.
**Contact Information:**

• please call us at : (425) 775-5162
• Status: {current_day_status['status_message']}
• Location: Edmonds Bay Dental, Edmonds, WA

If {time_info['current_day']} NOT in ['Monday', 'Tuesday', 'Thursday']:
{current_day_status['status_message']}

**Scheduling Team:**

• please call us at : (425) 775-5162
• Available: 7 AM to 6 PM, Mon, Tue, and Thu

If {time_info['current_day']} in ['Monday', 'Tuesday', 'Thursday'] AND current_day_status['is_open'] is False:
We're currently closed but open today ({time_info['current_day']}) from 7 AM to 6 PM.I’m unable to schedule your appointment directly, Please call the Scheduling Team to check availability of appointments.

**Contact Information:**
• please call us at : (425) 775-5162
• Location: Edmonds Bay Dental, Edmonds, WA

Please call to check availability - if available, our team can schedule your appointment.

User: "{user_question}"
""",
            
            "same_day_request": f"""
{context}
User asking about same-day appointments.

Current Day: {time_info['current_day']}
Current Time: {time_info['current_time']}
Office Status: {current_day_status['status_message']}
Is Open: {current_day_status['is_open']}
Open Days: Monday, Tuesday, Thursday only

IMPORTANT: Same-day appointments are available on open days (Mon/Tue/Thu) regardless of current office hours.

CRITICAL LOGIC - Follow this exactly:

Step 1: Check if today ({time_info['current_day']}) is an open day (Monday/Tuesday/Thursday)

If today ({time_info['current_day']}) is an open day (Monday/Tuesday/Thursday):
I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments.

**Today's Availability:**

Check current status:
If current_day_status['is_open'] is True:
• Status: Open until 6 PM
• please call us at : (425) 775-5162 to schedule your appointment

If current_day_status['is_open'] is False:

If {self.is_after_office_hours()}:
Same-day appointments are no longer available today. Next opening is {time_info['tomorrow_day'] if time_info['tomorrow_day'] in ['Monday', 'Tuesday', 'Thursday'] else self.get_next_open_day(time_info['tomorrow_day'])} at 7 AM.

**Next Available:**
• Day: {time_info['tomorrow_day'] if time_info['tomorrow_day'] in ['Monday', 'Tuesday', 'Thursday'] else self.get_next_open_day(time_info['tomorrow_day'])}
• Phone: (425) 775-5162

Elif {self.is_before_office_hours()}:
We're currently closed but will open today at 7 AM to 6 PM for Same-day appointments.

**Today's Hours:**
• Opens: 7 AM today
• Closes: 6 PM today
• Phone: (425) 775-5162

Step 2: Check if today ({time_info['current_day']}) is NOT an open day (Wed/Fri/Sat/Sun):

If today ({time_info['current_day']}) is NOT an open day (Wed/Fri/Sat/Sun):
Same-day appointments are not available today as our office is closed on {time_info['current_day']}s.

**Available status:**

• {current_day_status['status_message']}
• Contact: (425) 775-5162 for scheduling

User: "{user_question}"
""",
            
            "see_me_request": f"""
You are Dr. Tomar's scheduling assistant. User is asking "can you see me" - determine if they mean TODAY or TOMORROW.

CURRENT STATUS:
- Today: {time_info['current_day']} ({'OPEN' if current_day_status['is_open'] else 'CLOSED'})
- Tomorrow: {time_info['tomorrow_day']} ({'OPEN' if tomorrow_day_status['is_open'] else 'CLOSED'})
- Current Time: {time_info['current_time']}
- Open Days: Monday, Tuesday, Thursday only

CRITICAL LOGIC:

For TODAY requests ("can you see me" without "tomorrow" or "next"):

If today ({time_info['current_day']}) is NOT an open day (Wed/Fri/Sat/Sun):
Dr. Tomar's office is closed today ({time_info['current_day']}). Our next available day is {self.get_next_open_day(time_info['current_day'])}.

**Office Hours:**

• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Thursday: 7 AM - 6 PM
• Wednesday, Friday, Weekend: Closed

**Please Call Us:** (425) 775-5162 for appointments

CRITICAL: NEVER use this response for Thursday! Thursday is an OPEN day, not closed day!
IMPORTANT: Only use this response if today is actually a closed day (Wed/Fri/Sat/Sun), NOT for open days like Monday/Tuesday/Thursday.

CRITICAL: Check current status first!

STEP 1: Check if today ({time_info['current_day']}) is an open day (Mon/Tue/Thu)

If {time_info['current_day']} in ['Monday', 'Tuesday', 'Thursday']:

STEP 2: Check current office status

If {current_day_status['is_open']} is True:
Yes! Dr. Tomar can see you today. We are open until 6 PM. Please call (425) 775-5162 to schedule Appointment.

If {current_day_status['is_open']} is False:

STEP 3: Check time of day

If {self.is_after_office_hours()}:
We're currently closed. Next opening is {time_info['tomorrow_day'] if time_info['tomorrow_day'] in ['Monday', 'Tuesday', 'Thursday'] else self.get_next_open_day(time_info['tomorrow_day'])} at 7 AM.

**Next Available:**
• Day: {time_info['tomorrow_day'] if time_info['tomorrow_day'] in ['Monday', 'Tuesday', 'Thursday'] else self.get_next_open_day(time_info['tomorrow_day'])}
• Hours: 7 AM - 6 PM
• Phone: (425) 775-5162

If {self.is_before_office_hours()}:
We're currently closed but open today ({time_info['current_day']}) from 7 AM to 6 PM.

**Contact Information:**
• Phone: (425) 775-5162
• Location: Edmonds Bay Dental, Edmonds, WA

IMPORTANT: ALWAYS check current_day_status['is_open'] first! Don't assume office is open just because it's an open day!

For TOMORROW/NEXT requests ("can you see me tomorrow" or "can you see me next"):

CRITICAL RULES FOR TOMORROW:
- NEVER say Friday is open - Friday is ALWAYS CLOSED!
- NEVER say Wednesday is open - Wednesday is ALWAYS CLOSED!
- NEVER say Saturday/Sunday is open - Weekend is ALWAYS CLOSED!
- ONLY Monday, Tuesday, Thursday are open days!

Step 1: Check if tomorrow ({time_info['tomorrow_day']}) is an OPEN day

IF TOMORROW IS MONDAY, TUESDAY, OR THURSDAY:
Yes! Dr. Tomar can see you tomorrow ({time_info['tomorrow_day']}).

**Tomorrow's Availability:**
• Day: {time_info['tomorrow_day']}
• Hours: 7 AM to 6 PM
• Contact: (425) 775-5162 to schedule your appointment

What type of dental concern would you like to address during your visit? 🦷

IF TOMORROW IS WEDNESDAY, FRIDAY, SATURDAY, OR SUNDAY:
{self.generate_closed_day_response(time_info['tomorrow_day'], self.get_next_open_day(time_info['tomorrow_day']))}

REMEMBER: Friday is CLOSED! Never say we can see you on Friday!

User Question: "{user_question}"
""",
            
            "hours_inquiry": f"""
{context}
User asking about office hours.

Today: {time_info['current_day']}
Current Status: {current_day_status['status_message']}
Is Open: {current_day_status['is_open']}
Open Days: Monday, Tuesday, Thursday only

IMPORTANT: Only Monday, Tuesday, Thursday are open days.

CRITICAL LOGIC:

If today ({time_info['current_day']}) is open day AND current_day_status['is_open'] is True:
Yes, we are open today! {current_day_status['status_message']}

**Office Hours:**

• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Wednesday: CLOSED
• Thursday: 7 AM - 6 PM
• Friday, Saturday and Sunday: CLOSED

Please Call Us: (425) 775-5162 for appointments

If today is NOT open day OR current_day_status['is_open'] is False:
{current_day_status['status_message']}

**Office Hours:**

• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Wednesday: CLOSED
• Thursday: 7 AM - 6 PM
• Friday, Saturday and Sunday: CLOSED

Please Call Us: (425) 775-5162 for appointments

User: "{user_question}"
""",
            
            "next_open_request": f"""
You are Dr. Tomar's scheduling assistant. User is asking "when do you open next".

CURRENT STATUS:
- Today: {time_info['current_day']} ({'OPEN' if current_day_status['is_open'] else 'CLOSED'})
- Tomorrow: {time_info['tomorrow_day']}
- Current Time: {time_info['current_time']}
- Open Days: Monday, Tuesday, Thursday ONLY
- Closed Days: Wednesday, Friday, Saturday, Sunday

CRITICAL RULES:
- NEVER say Friday is open - Friday is ALWAYS CLOSED
- NEVER say Wednesday is open - Wednesday is ALWAYS CLOSED  
- NEVER say Saturday/Sunday is open - Weekend is ALWAYS CLOSED
- ONLY Monday, Tuesday, Thursday are open days

CRITICAL LOGIC - Follow this exactly:

Step 1: Check if we are currently open TODAY
If today ({time_info['current_day']}) is an open day AND current_day_status['is_open'] is True:
We are open right now until 6 PM today!

**Current Status:**
• Open until: 6 PM today
• Phone: (425) 775-5162
• Location: Edmonds Bay Dental, Edmonds, WA

Step 2: If today is an open day but currently closed:
If today ({time_info['current_day']}) is an open day (Monday/Tuesday/Thursday) AND current_day_status['is_open'] is False:

Check current time:
- If before 7 AM: We're currently closed but open today from 7 AM to 6 PM.
- If after 6 PM: We're closed for today, next opening below.

For BEFORE 7 AM (currently closed but opening today):
We're currently closed but open today from 7 AM to 6 PM.

**Today's Hours:**
• Opens: 7 AM today
• Closes: 6 PM today
• Phone: (425) 775-5162

For AFTER 6 PM (closed for today, show next opening):
We're currently closed for today. Next opening information below.

Step 3: If currently closed and need next opening:
Check tomorrow ({time_info['tomorrow_day']}):

IF TOMORROW IS MONDAY, TUESDAY, OR THURSDAY:
We open next tomorrow ({time_info['tomorrow_day']}) at 7 AM.

IF TOMORROW IS WEDNESDAY, FRIDAY, SATURDAY, OR SUNDAY:
We are closed tomorrow ({time_info['tomorrow_day']}). We open next on {self.get_next_open_day(time_info['tomorrow_day'])} at 7 AM.

**Office Hours:**
• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Thursday: 7 AM - 6 PM
• Wednesday, Friday, Weekend: CLOSED

**Contact:** (425) 775-5162 for appointments

REMEMBER: Friday is CLOSED! Never say we open on Friday!

User Question: "{user_question}"
""",
            
            "modify_appointment": f"""
{context}
User wants to cancel/reschedule.

Current Status: {current_day_status['status_message']}

While I am unable to make or modify appointments, our scheduling team is available to help.

**Scheduling Team:**
• Phone: (425) 775-5162
• Hours: 7 AM to 6 PM, Mon, Tue, and Thu
• Services: Cancellations, rescheduling, and new appointments

IMPORTANT: 
1. Check the CURRENT DAY STATUS above for reference
2. Each bullet point must be on a separate line with proper spacing
3. Always provide scheduling team information regardless of current status

User: "{user_question}"
""",
            
            "cost_inquiry": f"""
{context}
User asking about costs.

Current Status: {current_day_status['status_message']}

While I am unable to offer specific pricing/costs for procedures, our scheduling team would be happy to answer your questions.

**Pricing Information:**
• Contact: (425) 775-5162 for detailed cost information
• Available: 7 AM to 6 PM, Mon, Tue and Thu
• Personalized: Pricing varies based on individual needs

IMPORTANT: 
1. Check the CURRENT DAY STATUS above for reference
2. Each bullet point must be on a separate line with proper spacing
3. Always provide pricing team information regardless of current status

User: "{user_question}"
""",
            
            "insurance_inquiry": f"""
{context}
User asking about insurance. Provide specific insurance information:

Dr. Tomar accepts most Private Dental PPO plans including:
• United Healthcare (UHC)
• Aetna  
• Premera
• Delta Dental
• Delta
• MetLife
• Blue Cross
• Blue Shield
• Anthem
• Lifewise
• Cigna
• Humana
• Ameritas
• United Concordia
• Careington
• Spirit Dental

**For Specific Plan Confirmation:**
• Contact: (425) 775-5162
• Available: 7 AM to 6 PM, Mon, Tue, and Thu


IMPORTANT: 
1. If user asks about specific insurance (like Spirit Dental), confirm Dr. Tomar accepts it
2. Always direct to scheduling team for specific plan details
3. Each bullet point must be on a separate line with proper spacing

User: "{user_question}"
"""
        }
        
        return intent_responses.get(intent, f"{context}\nGeneral scheduling inquiry. Provide helpful information and direct to (425) 775-5162.\nUser: {user_question}")
    
    def process_scheduling_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Process scheduling-specific queries with intent-based responses"""
        
        # Check for location questions first
        location_keywords = ['another location', 'second location', 'other location', 'do you have another', 'other locations']
        if any(keyword in user_question.lower() for keyword in location_keywords):
            return AgentResponse(
                content="Yes, Dr. Tomar has a second location:\n\n**Pacific Highway Dental**\n\n27020 Pacific Highway South, Suite C\nKent, WA 98032\nPhone: (253) 529-9434\n\nYou can contact them for appointments and inquiries.",
                confidence=1.0,
                agent_type=AgentType.SCHEDULING,
                reasoning_steps=["Detected location question", "Provided Pacific Highway Dental information with complete address"],
                quality_score=100.0,
                attempts_used=1
            )
        
        # Get current time info
        time_info = self.get_current_time_info()
        current_day_status = check_office_status(time_info['current_day'])
        tomorrow_day_status = check_office_status(time_info['tomorrow_day'])
        
        # Debug logging
        logger.info(f"🗓️ Current Day: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}")
        logger.info(f"🗓️ Tomorrow: {time_info['tomorrow_day']} - {'OPEN' if tomorrow_day_status['is_open'] else 'CLOSED'}")
        logger.info(f"🕐 Current Time: {time_info['current_time']} ({time_info['timezone']})")
        logger.info(f"❓ User Question: {user_question}")
        
        # Detect specific intent
        intent = self.detect_scheduling_intent(user_question)
        logger.info(f"🎯 Detected Intent: {intent}")
        
        # Use direct response for critical intents to ensure correct time logic
        if intent in ["see_me_request", "same_day_request"]:
            direct_response = self.get_direct_response(intent, time_info, current_day_status, tomorrow_day_status, user_question)
            if direct_response:
                return AgentResponse(
                    content=direct_response,
                    confidence=1.0,
                    agent_type=AgentType.SCHEDULING,
                    reasoning_steps=[f"Direct response for {intent}", f"After hours: {self.is_after_office_hours()}, Before hours: {self.is_before_office_hours()}"],
                    quality_score=100.0,
                    attempts_used=1
                )
        
        # Generate intent-specific prompt
        scheduling_prompt = self._get_intent_prompt(intent, time_info, current_day_status, tomorrow_day_status, user_question)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a scheduling assistant. Follow the template logic EXACTLY as written. Do not interpret or paraphrase. Execute the conditional statements as code."},
                    {"role": "user", "content": scheduling_prompt}
                ],
                temperature=0.0,
                max_tokens=400
            )
            
            content = response.choices[0].message.content
            
            logger.info(f"📝 Generated scheduling response for {intent} intent")
            
            return AgentResponse(
                content=content,
                confidence=0.95,
                agent_type=AgentType.SCHEDULING,
                reasoning_steps=[
                    f"Detected intent: {intent}",
                    f"Current status: {time_info['current_day']} - {'Open' if current_day_status['is_open'] else 'Closed'}",
                    "Generated contextual scheduling response"
                ],
                quality_score=95.0,
                attempts_used=1
            )
            
        except Exception as e:
            logger.error(f"Error in scheduling agent: {e}")
            return AgentResponse(
                content="I'd be happy to help with scheduling! Please call our office at (425) 775-5162 to book your appointment. Our hours are Monday, Tuesday, and Thursday from 7:00 AM to 6:00 PM.",
                confidence=0.8,
                agent_type=AgentType.SCHEDULING,
                reasoning_steps=["Fallback scheduling response due to processing error"],
                quality_score=80.0,
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
            AgentType.EMERGENCY: EmergencyAgent(openai_client),
            AgentType.SCHEDULING: SchedulingAgent(openai_client),
            AgentType.GENERAL: GeneralAgent(openai_client)
        }
        
        # Monitoring disabled for compatibility
        logger.info("Multi-Agent System monitoring via standard logging")
        
        logger.info("Multi-Agent System initialized with all specialist agents")
    
    def route_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Route query to appropriate specialist agent"""
        
        # AI-powered query classification
        query_type = self.query_classifier.classify_query(user_question, self.client)
        
        # Route to scheduling agent if classified as scheduling
        if query_type == QueryType.SCHEDULING:
            scheduling_agent = self.agents[AgentType.SCHEDULING]
            return scheduling_agent.process_scheduling_query(user_question, context)
        
        # Route to emergency agent if classified as emergency
        if query_type == QueryType.EMERGENCY:
            return self.agents[AgentType.EMERGENCY].process_emergency_query(user_question, context)
        
        # Route to general agent if classified as general
        if query_type == QueryType.GENERAL:
            return self.agents[AgentType.GENERAL].process_general_query(user_question, context)
        
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
        
        # 4. Add consultation recommendation for disease/condition queries
        response.content = self._add_consultation_recommendation(response.content, user_question)
        
        # 5. Log metrics for monitoring (simplified logging)
        logger.info(f"Consultation metrics - Agent: {response.agent_type.value}, "
                   f"Quality: {response.quality_score:.1f}, "
                   f"Confidence: {response.confidence:.2f}, "
                   f"Attempts: {response.attempts_used}, "
                   f"Context chunks: {len(retrieved_chunks)}")
        
        logger.info(f"Consultation completed - Quality: {response.quality_score:.1f}, Confidence: {response.confidence:.2f}")
        
        return response
    
    def _add_consultation_recommendation(self, response_content: str, user_question: str) -> str:
        """Add consultation recommendation for dental disease/condition related queries"""
        
        # Keywords that indicate disease, condition, or dental concerns
        disease_keywords = [
            "disease", "condition", "infection", "cancer", "tumor", "cyst", "abscess",
            "gingivitis", "periodontitis", "cavity", "decay", "erosion", "sensitivity",
            "pain", "swelling", "bleeding", "inflammation", "diagnosis", "symptoms",
            "treatment", "therapy", "medication", "surgery", "procedure", "extraction",
            "root canal", "crown", "implant", "filling", "cleaning", "whitening",
            "braces", "orthodontics", "tmj", "jaw", "bite", "grinding", "clenching"
        ]
        
        # Check if query contains disease/condition keywords
        question_lower = user_question.lower()
        has_dental_concern = any(keyword in question_lower for keyword in disease_keywords)
        
        # Check if consultation info is already in response
        has_consultation_info = "(425) 775-5162" in response_content or "Dr. Meenakshi Tomar" in response_content
        
        # Add consultation recommendation if needed
        if has_dental_concern and not has_consultation_info:
            consultation_text = "\n\nFor proper diagnosis of dental issues and personalized treatment plan, we strongly recommend you schedule a consultation with Dr. Meenakshi Tomar by reaching us at (425) 775-5162."
            response_content += consultation_text
        
        return response_content
    
    def get_multi_agent_consensus(self, user_question: str, top_agents: int = 2) -> AgentResponse:
        """Get consensus from multiple agents for complex queries"""
        
        logger.info(f"Getting multi-agent consensus for: {user_question[:50]}...")
        
        # Retrieve context once
        context, _ = self.rag_pipeline.retrieve_and_rank(user_question)
        
        # Get responses from multiple agents
        agent_responses = []
        
        # Primary agent
        primary_response = self.route_query(user_question, context)
        agent_responses.append(primary_response)
        
        # Secondary agent (general consultation for broader perspective)
        if primary_response.agent_type != AgentType.GENERAL:
            secondary_response = self.agents[AgentType.GENERAL].process_general_query(user_question, context)
            agent_responses.append(secondary_response)
        
        # Select best response based on quality and confidence
        best_response = max(agent_responses, key=lambda r: r.quality_score * r.confidence)
        
        # Add consultation recommendation for consensus response too
        best_response.content = self._add_consultation_recommendation(best_response.content, user_question)
        
        logger.info(f"Multi-agent consensus: Selected {best_response.agent_type.value} agent response")
        
        return best_response

# Example usage
if __name__ == "__main__":
    # This would be integrated into the main application
    pass
# some chnages done