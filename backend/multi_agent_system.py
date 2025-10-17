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
- Dr. Tomar focuses on overall patient wellbeing and care continuity
- Dr. Tomar welcomes patients from all locations and states
- Expert in explaining consultation processes for out-of-state patients
- Skilled in providing general dental information and office policies
- Dr. Tomar only gives dental-related solutions, not out-of-context advice
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

class EmergencyAgent(BaseAgent):
    """Simple emergency agent with fast responses"""
    
    def __init__(self, openai_client):
        super().__init__(openai_client, AgentType.EMERGENCY)
        
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
    
    def get_emergency_advice(self, user_question: str) -> str:
        """Get dynamic emergency advice based on user's specific condition"""
        q = user_question.lower()
        advice_parts = []
        
        # Broken/damaged teeth
        if any(word in q for word in ['broke', 'broken', 'chipped', 'cracked', 'fractured']):
            advice_parts.append("**For Broken/Chipped Tooth:**\n• Rinse mouth gently with warm water\n• Save any broken pieces in milk or saliva\n• Apply cold compress to reduce swelling\n• Avoid chewing on the affected side")
        
        # Pain management
        if any(word in q for word in ['pain', 'hurt', 'ache', 'throbbing', 'severe', 'unbearable']):
            if any(word in q for word in ['severe', 'unbearable', 'excruciating']):
                advice_parts.append("**For Severe Pain:**\n• Take over-the-counter pain reliever (follow package directions)\n• Apply cold compress for 15-20 minutes\n• Rinse with warm salt water (1/2 tsp salt in warm water)\n• Avoid very hot or cold foods/drinks\n• Keep head elevated when lying down")
            else:
                advice_parts.append("**For Tooth Pain:**\n• Rinse with warm salt water\n• Take pain reliever as directed\n• Apply cold compress to outside of cheek\n• Avoid hard, sticky, or very hot/cold foods")
        
        # Swelling
        if any(word in q for word in ['swollen', 'swelling', 'puffy', 'inflamed']):
            advice_parts.append("**For Swelling:**\n• Apply cold compress for 15-20 minutes, then remove for 15 minutes\n• Keep head elevated when resting\n• Avoid hot foods and drinks\n• Rinse gently with salt water")
        
        # Bleeding
        if any(word in q for word in ['bleeding', 'blood', 'hemorrhage']):
            advice_parts.append("**For Bleeding:**\n• Apply gentle pressure with clean gauze or cloth\n• Rinse very gently with cold water\n• Avoid spitting, rinsing vigorously, or using straws\n• If bleeding doesn't stop in 15 minutes, seek immediate care")
        
        # Knocked out tooth
        if any(phrase in q for phrase in ['knocked out', 'fell out', 'lost tooth', 'tooth came out']):
            advice_parts.append("**For Knocked Out Tooth (URGENT):**\n• Handle tooth by crown only, not the root\n• Rinse gently if dirty (don't scrub)\n• Try to reinsert in socket if possible\n• If not possible, keep in milk or saliva\n• Get to dentist within 30 minutes for best chance of saving tooth")
        
        # Abscess/infection
        if any(word in q for word in ['abscess', 'infection', 'pus', 'fever']):
            advice_parts.append("**For Possible Infection:**\n• Rinse with warm salt water several times daily\n• Apply cold compress to reduce swelling\n• Take pain reliever as needed\n• Do NOT apply heat or hot compress\n• Seek immediate care if fever develops")
        
        # Object stuck in teeth
        if any(phrase in q for phrase in ['stuck', 'lodged', 'trapped', 'food stuck']):
            advice_parts.append("**For Object Stuck in Teeth:**\n• Try gentle flossing to remove\n• Rinse with warm water\n• Do NOT use sharp objects like pins or needles\n• If unable to remove, see dentist promptly")
        
        # Jaw injury
        if any(word in q for word in ['jaw', 'tmj', 'locked', 'dislocated']):
            advice_parts.append("**For Jaw Injury:**\n• Apply cold compress to reduce swelling\n• Eat only soft foods\n• Avoid opening mouth wide\n• Support jaw with bandage if necessary\n• Seek immediate dental care")
        
        # General emergency advice if no specific condition detected
        if not advice_parts:
            advice_parts.append("**General Emergency Care:**\n• Rinse mouth gently with warm water\n• Apply cold compress if swelling\n• Take over-the-counter pain reliever if needed\n• Avoid hard, sticky, or extreme temperature foods\n• Contact dentist as soon as possible")
        
        return "\n\n".join(advice_parts)
    
    def detect_emergency_intent(self, user_question: str) -> str:
        """Detect emergency intent"""
        q = user_question.lower()
        
        if any(word in q for word in ['tomorrow', 'next']):
            return 'tomorrow_emergency'
        else:
            return 'today_emergency'
    
    def generate_emergency_response(self, intent: str, time_info: Dict, user_question: str) -> str:
        """Generate emergency responses"""
        current_day = time_info['current_day']
        tomorrow_day = time_info.get('tomorrow_day', '')
        hour = time_info['hour']
        next_open = self.get_next_open_day(current_day)
        advice = self.get_emergency_advice(user_question)
        
        # Check office status
        is_open_day = current_day in ['Monday', 'Tuesday', 'Thursday']
        is_office_hours = 7 <= hour < 18
        is_open = is_open_day and is_office_hours
        is_tomorrow_open = tomorrow_day in ['Monday', 'Tuesday', 'Thursday']
        
        if intent == 'tomorrow_emergency':
            if is_tomorrow_open:
                return f"Yes, Dr. Tomar can see you for emergency tomorrow ({tomorrow_day}) 7 AM-6 PM. Call: (425) 775-5162 to schedule emergency appointment."
            else:
                next_after_tomorrow = self.get_next_open_day(tomorrow_day)
                base_response = f"Dr. Tomar's office is closed tomorrow ({tomorrow_day}). Emergency care: Call (425) 775-5162. Next open: {next_after_tomorrow} 7 AM-6 PM."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
        
        else:  # today_emergency
            if not is_open_day:
                base_response = f"Dr. Tomar's office is closed today ({current_day}). Emergency care: Call (425) 775-5162. Next open: {next_open} 7 AM-6 PM."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
            elif is_open:
                return "Dr. Tomar's office is currently open for emergency appointments.I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments. \n\n**Status:**\n\n• Open until 6 PM today.\n please Call us at : (425) 775-5162 to schedule your appointment"
            elif hour < 7:
                base_response = f"Currently closed but we open today at 7 AM to 6 PM for emergency care. Call: (425) 775-5162."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
            else:  # hour >= 18
                base_response = f"Currently closed (after 6 PM). Emergency care: Call (425) 775-5162. Next open: {next_open} 7 AM-6 PM."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
    
    def process_emergency_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Process emergency queries"""
        
        # Get time info and detect intent
        time_info = self.get_current_time_info()
        intent = self.detect_emergency_intent(user_question)
        content = self.generate_emergency_response(intent, time_info, user_question)
        
        return AgentResponse(
            content=content,
            confidence=0.95,
            agent_type=AgentType.EMERGENCY,
            reasoning_steps=[f"Emergency intent: {intent}", f"Day: {time_info['current_day']}"],
            quality_score=95.0,
            attempts_used=1
        )

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
        elif any(word in q for word in ['cancel', 'reschedule','modified', 'change']):
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
                return f"Same-day appointments not available today ({current_day} - office closed). **Next available **: {next_open} 7 AM-6 PM.• Open Hours: 7 AM to 6 PM •please call us at : (425) 775-5162 to schedule your appointment"
                
                
            elif is_open:
                return "Dr Tomar Clinic Open , Call now to schedule: (425) 775-5162.I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments. **Today's Availability:** • Status: Open until 6 PM • Please call us at : (425) 775-5162 to schedule your appointment"
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM for same-day appointments. I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments. **Today's Availability:** • Opens at 7 AM -6 PM • please call us at : (425) 775-5162 to schedule your appointment"
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Next available: {next_open} 7 AM-6 PM.I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments. **Next's Availability:** • Opens at 7 AM -6 PM • please call us at : (425) 775-5162 to schedule your appointment"
        
        elif intent == 'see_me_request':
            if not is_open_day:
                return f"Dr. Tomar's office is closed today ({current_day}). Next availability: {next_open} 7 AM-6 PM.\n\n**Office Hours:**\n\n• Monday: 7 AM - 6 PM\n• Tuesday: 7 AM - 6 PM\n• Thursday: 7 AM - 6 PM\n• Wednesday, Friday, Weekend: Closed\n\n**Please Call Us:** (425) 775-5162 for appointments"
            elif is_open:
                return "Dr. Tomar clinic open. Please Call now: (425) 775-5162. Currently open until 6 PM.\n\n**Office Hours:**\n\n• Monday: 7 AM - 6 PM\n• Tuesday: 7 AM - 6 PM\n• Thursday: 7 AM - 6 PM\n• Wednesday, Friday, Weekend: Closed\n\n**Please Call Us:** (425) 775-5162 for appointments"
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Please Call: (425) 775-5162 to check availability.\n\n**Office Hours:**\n\n• Monday: 7 AM - 6 PM\n• Tuesday: 7 AM - 6 PM\n• Thursday: 7 AM - 6 PM\n• Wednesday, Friday, Weekend: Closed\n\n**Please Call Us:** (425) 775-5162 for appointments"
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Next availability: {next_open} 7 AM-6 PM.\n\n**Office Hours:**\n\n• Monday: 7 AM - 6 PM\n• Tuesday: 7 AM - 6 PM\n• Thursday: 7 AM - 6 PM\n• Wednesday, Friday, Weekend: Closed\n\n**Please Call Us:** (425) 775-5162 for appointments"
        
        elif intent == 'hours_inquiry':
            if not is_open_day:
                return f"Office closed today ({current_day}). Next open: {next_open} 7 AM-6 PM. Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Closed: Wed/Fri/Weekend. Please Call: (425) 775-5162"
            elif is_open:
                return "Currently open until 6 PM today! Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Closed: Wed/Fri/Weekend.Please Call: (425) 775-5162 for Scheduling your appointment."
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM.  Please Call: (425) 775-5162"
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Office Hours: Monday/Tuesday/Thursday 7 AM-6 PM. Next open: {next_open} 7 AM-6 PM. Call: (425) 775-5162"
        
        elif intent == 'modify_appointment':
            if not is_open_day:
                return f"Office closed today ({current_day}). To cancel/reschedule, call: (425) 775-5162. Next open: {next_open} 7 AM-6 PM"
            elif is_open:
                return "Clinic open ,Call now to cancel/reschedule: (425) 775-5162 Your Appointment. Currently open until 6 PM today."
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM. Please Call: (425) 775-5162 to cancel/reschedule your Appointment."
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). To cancel/reschedule, Please call: (425) 775-5162. Next open: {next_open} 7 AM-6 PM"
        
        elif intent == 'cost_inquiry':
            return "For pricing information, please call: (425) 775-5162. We'll discuss costs during consultation."
        
        elif intent == 'insurance_inquiry':
            return "For insurance coverage details, call: (425) 775-5162. We accept most major insurance plans."
        
        elif intent == 'tomorrow_request':
            if is_tomorrow_open:
                return f" we are open tomorrow ({tomorrow_day}) 7 AM-6 PM. Please Call (425) 775-5162 to schedule your appointment."
            else:
                next_after_tomorrow = self.get_next_open_day(tomorrow_day)
                return f"We are closed tomorrow ({tomorrow_day}). Next available: {next_after_tomorrow} 7 AM-6 PM.please Call: (425) 775-5162 to Schedule your apppointment"
        
        else:  # schedule_request
            if not is_open_day:
                return f"Office closed today ({current_day}). Call (425) 775-5162 to schedule. Next available: {next_open} 7 AM-6 PM.I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments.\n• Please call us at : (425) 775-5162 to schedule your appointment"
            elif is_open:
                return "Our clinic is open right now! I’m unable to schedule your appointment directly, but please give us a call and our team can book an appointment when a slot is available.\n\n**Contact Information:**\n\n • please call us at : (425) 775-5162 to schedule your appointment\n• Status: Open until 6 PM\n"
            elif hour < 7:
                return f"Currently closed but we open today at 7 AM to 6 PM.I’m unable to schedule your appointment directly, but please give us a call and our team can book an appointment when a slot is available.Please Call: (425) 775-5162 to schedule your appointment."
            else:  # hour >= 18
                return f"Currently closed (after 6 PM). Call (425) 775-5162 to schedule. Next available: {next_open} 7 AM-6 PM.I’m unable to schedule your appointment directly,Our team can schedule your appointment when available. Please call us at (425) 775-5162 to schedule your appointment."
    
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
            AgentType.EMERGENCY: EmergencyAgent(openai_client),
            AgentType.SCHEDULING: SchedulingAgent(openai_client),
            AgentType.GENERAL: BaseAgent(openai_client, AgentType.GENERAL)
        }
        
        logger.info("Multi-Agent System initialized with all specialist agents")
    
    def is_mixed_query(self, user_question: str) -> bool:
        """AI-powered mixed query detection"""
        try:
            mixed_detection_prompt = f"""
Analyze this dental consultation question and determine if it contains BOTH scheduling/appointment elements AND general information elements.

Question: "{user_question}"

Scheduling elements include: appointment booking, timing, availability, office hours, scheduling, canceling, rescheduling
General elements include: insurance, policies, procedures, health conditions, costs, locations, eligibility, COVID, illness

Respond with only "YES" if it contains BOTH elements, or "NO" if it contains only one type or neither.

Examples:
- "Can I get appointment tomorrow if I don't have insurance?" → YES (appointment + insurance)
- "Can I cancel my appointment if I test positive for COVID?" → YES (cancel appointment + COVID)
- "What are your office hours?" → NO (only scheduling)
- "Do you accept my insurance?" → NO (only general)

Answer:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": mixed_detection_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_mixed = result == "YES"
            
            # Debug logging
            print(f"Question: {user_question}")
            print(f"AI Mixed Detection: {result}")
            print(f"Is mixed: {is_mixed}")
            
            return is_mixed
            
        except Exception as e:
            print(f"Error in mixed query detection: {e}")
            # Fallback to simple keyword detection
            q = user_question.lower()
            has_scheduling = any(word in q for word in ['appointment', 'schedule', 'cancel', 'reschedule'])
            has_general = any(word in q for word in ['insurance', 'covid', 'cost', 'policy'])
            return has_scheduling and has_general
    
    def get_hybrid_response(self, user_question: str, context: str = "") -> AgentResponse:
        """Generate combined response from scheduling and general agents"""
        
        # Get scheduling response
        scheduling_agent = self.agents[AgentType.SCHEDULING]
        scheduling_response = scheduling_agent.process_scheduling_query(user_question, context)
        
        # Get general response
        general_agent = self.agents[AgentType.GENERAL]
        try:
            rag_context, _ = self.rag_pipeline.retrieve_and_rank(user_question)
        except:
            rag_context = ""
        general_response = general_agent.process_query(user_question, rag_context, QueryType.GENERAL)
        
        # Combine responses intelligently
        combined_content = self.combine_responses(scheduling_response.content, general_response.content, user_question)
        
        return AgentResponse(
            content=combined_content,
            confidence=(scheduling_response.confidence + general_response.confidence) / 2,
            agent_type=AgentType.SCHEDULING,  # Primary type
            reasoning_steps=scheduling_response.reasoning_steps + general_response.reasoning_steps,
            quality_score=(scheduling_response.quality_score + general_response.quality_score) / 2,
            attempts_used=max(scheduling_response.attempts_used, general_response.attempts_used)
        )
    
    def combine_responses(self, scheduling_content: str, general_content: str, user_question: str) -> str:
        """Intelligently combine scheduling and general responses"""
        
        # Extract key information from general response
        general_info = ""
        
        # Check for specific topics
        if "insurance" in user_question.lower():
            general_info = "Dr. Tomar accepts most major insurance plans including UHC, Aetna, Delta Dental, MetLife, and others. For specific coverage details, please call (425) 775-5162."
        
        elif "out of state" in user_question.lower() or "live out" in user_question.lower():
            general_info = "Dr. Tomar welcomes patients from all states and locations."
        
        elif "cost" in user_question.lower() or "price" in user_question.lower():
            general_info = "For accurate pricing information, please contact Dr. Tomar's office at (425) 775-5162. Costs vary based on individual needs and treatment complexity."
        
        elif "covid" in user_question.lower() or "coronavirus" in user_question.lower() or "test positive" in user_question.lower():
            general_info = "Yes, you can cancel or reschedule your appointment if you test positive for COVID-19 or have symptoms. Please call (425) 775-5162 to inform our team. We prioritize the health and safety of all patients and staff."
        
        elif "sick" in user_question.lower() or "illness" in user_question.lower() or "symptoms" in user_question.lower():
            general_info = "If you're feeling unwell or have symptoms, please call (425) 775-5162 to discuss rescheduling your appointment. We want to ensure the safety of all patients and staff."
        
        else:
            # For any other general topic, extract relevant info from general response
            if general_content and len(general_content.strip()) > 50:
                # Take first meaningful paragraph from general response
                lines = [line.strip() for line in general_content.split('\n') if line.strip()]
                if lines:
                    general_info = lines[0][:200] + "..." if len(lines[0]) > 200 else lines[0]
        
        # Combine with general info first, then scheduling
        if general_info:
            return f"{general_info}\n\n**Scheduling Information:**\n{scheduling_content}"
        else:
            return scheduling_content
    
    def route_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Route query to appropriate specialist agent"""
        
        # Check for mixed queries first
        if self.is_mixed_query(user_question):
            return self.get_hybrid_response(user_question, context)
        
        # AI-powered query classification
        query_type = self.query_classifier.classify_query(user_question, self.client)
        
        # Route to scheduling agent if classified as scheduling
        if query_type == QueryType.SCHEDULING:
            scheduling_agent = self.agents[AgentType.SCHEDULING]
            return scheduling_agent.process_scheduling_query(user_question, context)
        
        # Route to emergency agent if classified as emergency
        if query_type == QueryType.EMERGENCY:
            emergency_agent = self.agents[AgentType.EMERGENCY]
            return emergency_agent.process_emergency_query(user_question, context)
        
        # Route to appropriate agent
        agent = self.agents.get(query_type.value, self.agents[AgentType.GENERAL])
        
        # Get relevant context from RAG if needed
        if context:
            rag_context = context
        else:
            try:
                rag_context, _ = self.rag_pipeline.retrieve_and_rank(user_question)
            except:
                rag_context = ""
        
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
