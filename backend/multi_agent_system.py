"""
Multi-Agent System for Specialized Dental Consultation
Implements different specialist agents for various dental domains

Classifier + RAG Pipeline:
model = "gpt-4o-mini"

Final Answer Agent:
model = "gpt-4o-mini"
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
from openai import AsyncOpenAI
from loguru import logger
import asyncio
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
        # Add async client if available
        if hasattr(openai_client, 'api_key'):
            from openai import AsyncOpenAI
            self.async_client = AsyncOpenAI(api_key=openai_client.api_key)
        else:
            self.async_client = None
        
    def get_specialist_persona(self) -> str:
        """Get specialized persona for this agent type"""
        base_persona = self.cot_prompts._get_base_persona()
        
        specializations = {
            AgentType.DIAGNOSTIC: """
DIAGNOSTIC SPECIALIZATION:
You must always reply in the same language as the user's message.
Auto-detect the user's language and generate the response in that language only.
- Dr. Tomar is expert in symptom analysis and differential diagnosis
- Dr. Tomar is skilled in identifying urgent vs non-urgent conditions
- Dr. Tomar is experienced in pain assessment and oral pathology
- Dr. Tomar focuses on thorough symptom evaluation and risk assessment
""",
            AgentType.TREATMENT: """
TREATMENT SPECIALIZATION:
You must always reply in the same language as the user's message.
Auto-detect the user's language and generate the response in that language only.
- Dr. Tomar is expert in comprehensive treatment planning
- Dr. Tomar is skilled in explaining complex procedures clearly
- Dr. Tomar is experienced in treatment options and alternatives
- Dr. Tomar focuses on patient education and informed consent
""",
            AgentType.PREVENTION: """
PREVENTION SPECIALIZATION:
You must always reply in the same language as the user's message.
Auto-detect the user's language and generate the response in that language only.
- Dr. Tomar is expert in preventive dentistry and oral hygiene
- Dr. Tomar is skilled in patient education and behavior modification
- Dr. Tomar is experienced in risk factor assessment and management
- Dr. Tomar focuses on long-term oral health maintenance
""",
            AgentType.EMERGENCY: """
EMERGENCY SPECIALIZATION:
You must always reply in the same language as the user's message.
Auto-detect the user's language and generate the response in that language only.
- Dr. Tomar is expert in dental emergency assessment and triage
- Dr. Tomar is skilled in pain management and urgent care protocols
- Dr. Tomar is experienced in trauma and acute condition management
- Dr. Tomar focuses on immediate care and stabilization
""",
            AgentType.SCHEDULING: """
SCHEDULING SPECIALIZATION:
You must always reply in the same language as the user's message.
Auto-detect the user's language and generate the response in that language only.
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
You must always reply in the same language as the user's message.
Auto-detect the user's language and generate the response in that language only.
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
    
    async def process_query(
        self, 
        user_question: str, 
        context: str = "",
        query_type: QueryType = None,
        conversation_history: str = "",
        detected_lang: str = "English"
    ) -> AgentResponse:
        """Process query with specialized approach"""
        
        # Use pre-classified query type (no duplicate classification)
        agent_query_type = query_type or QueryType.GENERAL
        
        # Generate specialized prompt with conversation history
        specialist_persona = self.get_specialist_persona()
        cot_prompt = self.cot_prompts.get_chain_of_thought_prompt(
            agent_query_type, user_question, context, conversation_history
        )
        
        # Combine with LANGUAGE FIRST - most important instruction
        if detected_lang.lower() != "english":
            language_instruction = f"CRITICAL: User wrote in {detected_lang}. You MUST respond in {detected_lang} ONLY. Do NOT use English."
        else:
            language_instruction = "Respond in English as requested."
        
        # Add instruction to not repeat the question
        no_repeat_instruction = "IMPORTANT: Do NOT repeat or restate the user's question in your response. Start directly with your answer."
        
        full_prompt = f"{language_instruction}\n\n{no_repeat_instruction}\n\n{specialist_persona}\n\n{cot_prompt}"
        logger.info(f"🎯 Language instruction: {language_instruction}")
        
        # STREAMING MODE: Ultra fast response
        try:
            stream = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You MUST respond in {detected_lang}. This is mandatory."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                timeout=10,
                stream=True  # Enable streaming
            )
            
            # Collect streaming response
            initial_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    initial_response += chunk.choices[0].delta.content
            
            # Clean initial response first
            initial_response = self._remove_meta_commentary(initial_response)
            
            # Improve response through reprompting - DISABLED for speed optimization
            # final_response, attempts, quality_scores = self.reprompting_system.improve_response_with_reprompting(
            #     full_prompt, user_question, initial_response, context
            # )
            
            # Direct response for speed (3-5 sec faster)
            final_response = initial_response
            attempts = 1
            quality_scores = []
            
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
                base_response = f"Dr. Tomar's office is closed tomorrow ({tomorrow_day})\n.Emergency Care: Please call (425) 775-5162 for immediate assistance\n\n.The next available day is: {next_after_tomorrow} 7 AM-6 PM."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
        
        else:  # today_emergency
            if not is_open_day:
                base_response = f"Dr. Tomar's office is closed today ({current_day})\n.Emergency Care: Please call (425) 775-5162 for immediate assistance\n\n.The next available day is: {next_open} 7 AM-6 PM."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
            elif is_open:
                return "Dr. Tomar's office is currently open for emergency appointments.I’m unable to schedule your appointment directly, but our Scheduling Team can assist you with availability for same-day appointments. \n\n**Status:**\n\n• Open until 6 PM today.\n please Call us at : (425) 775-5162 to schedule your appointment"
            elif hour < 7:
                base_response = f"Currently closed but we open today at 7 AM to 6 PM \n.for emergency care. Call: (425) 775-5162 for book your appointment"
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
            else:  # hour >= 18
                base_response = f"Currently closed (after 6 PM)\n.Emergency Care: Please call (425) 775-5162 for immediate assistance\n\n.The next available day is: {next_open} 7 AM-6 PM."
                return base_response + (f"\n\n**Immediate Care Instructions:**\n\n{advice}" if advice else "")
    
    def generate_intelligent_emergency_response(self, user_question: str, intent: str, time_info: Dict) -> str:
        """Generate emergency responses based on intent only"""
        return self.generate_emergency_response(intent, time_info, user_question)
    
    def generate_basic_emergency_response(self, intent: str, time_info: Dict, user_question: str) -> str:
        """Fallback emergency response when AI generation fails"""
        current_day = time_info['current_day']
        hour = time_info['hour']
        is_open_day = current_day in ['Monday', 'Tuesday', 'Thursday']
        next_open = self.get_next_open_day(current_day)
        advice = self.get_emergency_advice(user_question)
        
        if not is_open_day:
            base_response = f"🚨 Emergency: Office closed today ({current_day}). Call (425) 775-5162. Next open: {next_open} 7 AM-6 PM."
        elif 7 <= hour < 18:
            base_response = "🚨 Emergency: Office open now until 6 PM. Call (425) 775-5162 immediately."
        else:
            base_response = f"🚨 Emergency: Office closed. Call (425) 775-5162. Next open: {next_open} 7 AM-6 PM."
        
        return base_response + (f"\n\n**Immediate Care:**\n{advice}" if advice else "")

    async def process_emergency_query_async(self, user_question: str, context: str = "", detected_lang: str = "English") -> AgentResponse:
        """Async version of emergency query processing"""
        
        # If language not detected, detect it here
        if detected_lang == "English" and self.async_client:
            try:
                lang_response = await self.async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Detect language. Reply only: English, Hindi, Spanish, etc."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                detected_lang = lang_response.choices[0].message.content.strip()
                logger.info(f"⚡ Emergency Agent Language Detection: {detected_lang}")
            except Exception as e:
                logger.warning(f"Language detection in emergency agent failed: {e}")
                detected_lang = "English"
        
        return self.process_emergency_query(user_question, context, detected_lang)
    
    def process_emergency_query(self, user_question: str, context: str = "", detected_lang: str = "English") -> AgentResponse:
        """Process emergency queries with intelligent responses"""
        
        # Get time info and detect intent
        time_info = self.get_current_time_info()
        intent = self.detect_emergency_intent(user_question)
        
        # Generate emergency response with language consideration
        content = self.generate_intelligent_emergency_response(user_question, intent, time_info)
        
        # If non-English language detected, add language instruction
        if detected_lang.lower() != "english":
            logger.info(f"🌍 Emergency response in {detected_lang}")
            # The AI will handle language in the response generation
        
        return AgentResponse(
            content=content,
            confidence=0.95,
            agent_type=AgentType.EMERGENCY,
            reasoning_steps=[f"Emergency intent: {intent}", f"Day: {time_info['current_day']}", f"Language: {detected_lang}"],
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
        elif any(word in q for word in ['tomorrow', 'next',"kl","kal","khule ho","open ho"]):
            return 'tomorrow_request'
        elif any(phrase in q for phrase in ['can you see me', 'see me']):
            return 'see_me_request'
        elif any(phrase in q for phrase in ['weekend', 'saturday', 'sunday', 'weekends']):
            return 'weekend_inquiry'
        elif any(word in q for word in ['hours', 'open', 'close', 'khule', 'band']):
            return 'hours_inquiry'
        elif any(word in q for word in ['cancel', 'reschedule','modified', 'change']):
            return 'modify_appointment'
        elif any(word in q for word in ['cost', 'price', 'fee']):
            return 'cost_inquiry'
        elif any(word in q for word in ['insurance', 'coverage']):
            return 'insurance_inquiry'
        elif any(phrase in q for phrase in ['mil sakti hai', 'le sakta hun', 'book kar sakte hain', 'appointment']):
            return 'schedule_request'
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
                return (f"Unfortunately,Our office is closed today ({current_day}). "
            f"The next available day is {next_open}, from 7 AM to 6 PM.\n\n"
             f"I’m unable to schedule appointments directly, but our scheduling team can assist you with same-day availability.\n\n" 
           
            f"📞 Please call us at (425) 775-5162 to schedule your appointment.")
                
            elif is_open:
                return (
            "Dr. Tomar’s Clinic is open today until 6 PM. "
            "I’m unable to schedule appointments directly, but our scheduling team can assist you with same-day availability.\n\n"
            "📞 Please call us at (425) 775-5162 to schedule your appointment."
)
            elif hour < 7:
                return (
            "Unfortunately,Our clinic is currently closed but will open today at 7 AM. "
            "Same-day appointments are available once we open. "
            "I’m unable to schedule appointments directly, but our scheduling team can assist.\n\n"
            "📞 Please call us at (425) 775-5162 to book your appointment.")  
            else:  # hour >= 18
                return (
            f"Unfortunately,Our office has closed for the day (after 6 PM). "
            f"The next available day is {next_open}, from 7 AM to 6 PM.\n\n"
            "📞 Please call us at (425) 775-5162 to schedule your appointment."
        )
  
        elif intent == 'see_me_request':

            if not is_open_day:
             return (
            f"Unfortunately,Dr. Tomar’s office is closed today ({current_day}). "
            f"The next available day is {next_open}, from 7 AM to 6 PM.\n\n"
            "**Open Office Hours:**\n"
            "• Monday: 7 AM - 6 PM\n"
            "• Tuesday: 7 AM - 6 PM\n"
            "• Thursday: 7 AM - 6 PM\n\n"
            
            "📞 Please call us at (425) 775-5162 for appointments."
        )

            elif is_open:
              return (
            "Dr. Tomar’s Clinic is open today until 6 PM.\n\n"
            "**Office Hours:**\n"
            "• Monday: 7 AM - 6 PM\n"
            "• Tuesday: 7 AM - 6 PM\n"
            "• Thursday: 7 AM - 6 PM\n"
            "• Wednesday, Friday, Weekend: Closed\n\n"
            "📞 Please call us at (425) 775-5162 to schedule your visit."
        )

            elif hour < 7:
             return (
            "Unfortunately,Our office is currently closed but will open today at 7 AM. "
            "You can call to check availability once we open.\n\n"
            "**Office Hours:**\n"
            "• Monday: 7 AM - 6 PM\n"
            "• Tuesday: 7 AM - 6 PM\n"
            "• Thursday: 7 AM - 6 PM\n\n"
            
            "📞 Please call us at (425) 775-5162 for appointments."
        )

            else:  # hour >= 18
              return (
            f"Our office has closed for the day (after 6 PM). "
            f"The next available day is {next_open}, from 7 AM to 6 PM.\n\n"
            "**Office Hours:**\n"
            "• Monday: 7 AM - 6 PM\n"
            "• Tuesday: 7 AM - 6 PM\n"
            "• Thursday: 7 AM - 6 PM\n"
            "• Wednesday, Friday, Weekend: Closed\n\n"
            "📞 Please call us at (425) 775-5162 for appointments."
        )

        elif intent == 'hours_inquiry':
        
          if not is_open_day:
           return (
            f"Unfortunately,Our office is closed today ({current_day}). "
            f"We will reopen on {next_open} from 7 AM to 6 PM.\n\n"
            "**Office Hours:** Monday, Tuesday, Thursday – 7 AM to 6 PM.\n\n"
           
            "📞 For more details, please call (425) 775-5162."
        )

          elif is_open:
           return (
            "We’re currently open until 6 PM today.\n\n"
            "**Regular Hours:** Monday, Tuesday, Thursday – 7 AM to 6 PM.\n"
            "Closed on Wednesday, Friday, and weekends.\n\n"
            "📞 Please call (425) 775-5162 to schedule your appointment."
        )

          elif hour < 7:
           return (
            "We’re currently closed but will open today at 7 AM.\n\n"
            "**Regular Hours:** Monday, Tuesday, Thursday – 7 AM to 6 PM.\n\n"
            
            "📞 Please call (425) 775-5162 for scheduling assistance."
        )

          else:  # hour >= 18
           return (
            f"Unfortunately,Our office has closed for the day (after 6 PM). "
            f"We’ll reopen on {next_open} from 7 AM to 6 PM.\n\n"
            "📞 For any scheduling needs, please call (425) 775-5162." )

        elif intent == 'modify_appointment':
          if not is_open_day:
           return (
            f"Unfortunately,Our office is closed today ({current_day}). "
            f"To reschedule or cancel your appointment, please call (425) 775-5162. "
            f"We will reopen on {next_open} from 7 AM to 6 PM."
        )

          elif is_open:
           return (
            "Our clinic is open today until 6 PM. "
            "To reschedule or cancel your appointment, please call (425) 775-5162."
        )

          elif hour < 7:
           return (
            "Unfortunately,We’re currently closed but will open today at 7 AM. "
            "You can call us at (425) 775-5162 to cancel or reschedule your appointment once we open."
        )

          else:  # hour >= 18
           return (
            f"Unfortunately,Our office has closed for the day (after 6 PM). "
            f"To reschedule or cancel, please call (425) 775-5162. "
            f"We’ll reopen on {next_open} from 7 AM to 6 PM."
        )

        elif intent == 'cost_inquiry':
          return (
        "For detailed pricing information, please contact our office at (425) 775-5162. "
        "Our team will be happy to discuss costs during your consultation."
    )

        elif intent == 'insurance_inquiry':
         return (
        "For insurance coverage information, please call (425) 775-5162. "
        "We’re happy to confirm whether your plan is accepted."
    )

        elif intent == 'weekend_inquiry':
         return (
        "Unfortunately, Dr. Tomar's office is closed on weekends (Saturday and Sunday). "
        "We're open Monday, Tuesday, and Thursday from 7 AM to 6 PM.\n\n"
        "📞 Please call (425) 775-5162 to schedule your appointment during our open days."
    )

        elif intent == 'tomorrow_request':
            if is_tomorrow_open:
                return (
                    f"Our office will be open tomorrow ({tomorrow_day}) from 7 AM to 6 PM. "
                    "📞 Please call (425) 775-5162 to schedule your appointment."
                )
            else:
                next_after_tomorrow = self.get_next_open_day(tomorrow_day)
                return (
                    f"Our office will be closed tomorrow ({tomorrow_day}). "
                    f"The next available day is {next_after_tomorrow}, from 7 AM to 6 PM.\n\n"
                    "📞 Please call (425) 775-5162 to schedule your appointment."
                )

        else:  # schedule_request
            if not is_open_day:
                return (
                    f"Unfortunately,Our office is closed today ({current_day}). "
                    f"We’ll reopen on {next_open} from 7 AM to 6 PM. "
                    "I’m unable to schedule appointments directly, but our scheduling team can assist you.\n\n"
                    "📞 Please call (425) 775-5162 to book your appointment."
                )
            elif is_open:
                return (
                    "Our clinic is open right now until 6 PM. "
                    "I’m unable to schedule appointments directly, but our team will gladly help you find the next available slot.\n\n"
                    "📞 Please call (425) 775-5162 to schedule your appointment."
                )
            elif hour < 7:
                return (
                    "Unfortunately,We’re currently closed but will open today at 7 AM. "
                    "Our team will be happy to assist with booking once we open.\n\n"
                    "📞 Please call (425) 775-5162 to schedule your appointment."
                )
            else:  # hour >= 18
                return (
                    f"Our office has closed for the day (after 6 PM). "
                    f"The next available day is {next_open}, from 7 AM to 6 PM.\n\n"
                    "📞 Please call (425) 775-5162 to schedule your appointment."
                )
    async def generate_intelligent_response(self, user_question: str, intent: str, time_info: dict, detected_lang: str = "English") -> str:
        """Smart routing: Known intents = Fast, Unknown = AI Intelligence"""
        
        # Define known scheduling intents that have specific logic
        known_intents = [
            'same_day_request', 'tomorrow_request', 'hours_inquiry',
            'modify_appointment', 'cost_inquiry', 'insurance_inquiry', 
            'weekend_inquiry', 'see_me_request', 'schedule_request'
        ]
        
        if intent in known_intents:
            # Fast path: Use existing intent-based logic
            intent_response = self.generate_response(intent, time_info)
            return await self._improve_intent_response(user_question, intent_response, detected_lang)
        else:
            # Smart path: AI handles unknown scheduling questions
            return await self._generate_ai_scheduling_response(user_question, time_info, detected_lang)

    async def _improve_intent_response(self, user_question: str, intent_response: str, detected_lang: str = "English") -> str:
        """Improve existing intent responses to make them more natural"""
        try:
            prompt = f"""
MANDATORY: User asked in {detected_lang}. You MUST respond ONLY in {detected_lang}. DO NOT use English.

You are Dr. Meenakshi Tomar's Virtual assistant. Use the provided intent-based response as your base answer.

Patient Question: "{user_question}"
Intent-Based Response: "{intent_response}"

CRITICAL Instructions:
1. Use the Intent-Based Response as your main answer
2. Only improve formatting and make it more natural
3. Keep all office hours, phone numbers, and status information exactly as provided
4. DO NOT add greetings like "Hello" or "Thank you"
5. DO NOT mention current time in response
6. Make the response sound natural and conversational

Generate the final response using the intent-based answer:"""

            # STREAMING MODE for scheduling
            stream = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You MUST respond in {detected_lang}. This is mandatory."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5,
                stream=True
            )
            
            # Collect streaming response
            response_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            return response_text.strip()
        
        except Exception as e:
            # Fallback to intent-based response directly
            return intent_response

    async def _generate_ai_scheduling_response(self, user_question: str, time_info: dict, detected_lang: str = "English") -> str:
        """AI-powered response for scheduling questions not covered by intents"""
        
        current_day = time_info['current_day']
        hour = time_info['hour']
        is_open_day = current_day in ['Monday', 'Tuesday', 'Thursday']
        is_office_hours = 7 <= hour < 18
        is_open = is_open_day and is_office_hours
        next_open = self.get_next_open_day(current_day)
        
        # Build office status message
        if not is_open_day:
            office_status = f"CLOSED today ({current_day}). Next open: {next_open}"
        elif is_open:
            office_status = f"OPEN until 6 PM today ({current_day})"
        elif hour < 7:
            office_status = f"CLOSED now, opens today at 7 AM ({current_day})"  # Same day opening
        else:
            office_status = f"CLOSED for today (after 6 PM). Next open: {next_open}"
        
        try:
            prompt = f"""
MANDATORY: User asked in {detected_lang}. You MUST respond ONLY in {detected_lang}. DO NOT use English.

You are Dr. Meenakshi Tomar's virtual assistant. Answer this scheduling-related question professionally.

Patient Question: "{user_question}"

CURRENT OFFICE STATUS: {office_status}

OFFICE SCHEDULE:
- Open Days: Monday, Tuesday, Thursday (7 AM - 6 PM)
- Closed: Wednesday, Friday, Saturday, Sunday
- Today is: {current_day}
- Current time: {hour}:00 (24-hour format)
- Next open day: {next_open}

CRITICAL SCHEDULING RULES:
1. IMPORTANT: You CANNOT book appointments directly - only the scheduling team can book appointments
2. Always mention "I'm unable to schedule appointments directly, but our scheduling team can assist you"
3. For same-day appointment requests:
   - If today is CLOSED DAY (Wed/Fri/Weekend) → Say office is closed today, scheduling team available on {next_open}
   - If today is OPEN DAY but AFTER HOURS (after 6 PM) → Say closed for today, scheduling team available on {next_open}
   - If today is OPEN DAY and DURING HOURS (7 AM - 6 PM) → Say office is open, scheduling team available now
   - If today is OPEN DAY but BEFORE HOURS (before 7 AM) → Say opens today at 7 AM, scheduling team available then

4. For general questions (new patients, services, etc.):
   - Answer the question directly
   - Include relevant office information
   - Always mention scheduling team for appointments

OFFICE INFORMATION:
- Phone: (425) 775-5162
- We accept new patients
- We accept most major insurance plans
- Emergency appointments available
- Scheduling team handles all appointment bookings
- Second location: Pacific Highway Dental, Kent, WA

RESPONSE FORMATTING RULES:
- ALWAYS use line breaks (\n) to separate different parts of your response
- Put office status information on separate lines
- Add blank line (\n\n) before phone number for emphasis
- Break long sentences into multiple lines for better readability
- Use proper spacing between different topics

RESPONSE RULES:
- Always clarify you cannot book appointments directly
- Mention scheduling team availability based on office hours
- Always include phone number (425) 775-5162
- Be specific about office status when relevant
- IMPORTANT: If today is open day but before hours, mention "opens today at 7 AM" NOT next day
- Only mention next open day when today is completely closed (closed day or after hours)
- Don't mention current time explicitly in response
- Be professional and helpful

EXAMPLE RESPONSES WITH PROPER FORMATTING:
- "Yes, we accept new patients!\n\nI'm unable to schedule appointments directly, but our scheduling team can assist you.\n\n📞 Please call (425) 775-5162 to book your appointment."
- "Our office is currently open until 6 PM.\n\nI'm unable to schedule appointments directly, but our scheduling team is available now.\n\n📞 Please call (425) 775-5162 to book your appointment."
- "Our office is closed today ({current_day}).\n\nI'm unable to schedule appointments directly, but our scheduling team will be available on {next_open} from 7 AM to 6 PM.\n\n📞 Please call (425) 775-5162 to book your appointment."
- "Our office opens today at 7 AM.\n\nI'm unable to schedule appointments directly, but our scheduling team will be available once we open.\n\n📞 Please call (425) 775-5162 to book your appointment."

IMPORTANT: Format your response with proper line breaks (\n) and spacing for better user visibility. Separate different information into new lines.

Generate response:"""

            # STREAMING MODE for AI scheduling
            stream = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You MUST respond in {detected_lang}. This is mandatory."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
                stream=True
            )
            
            # Collect streaming response
            response_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            return response_text.strip()
        
        except Exception as e:
            # Fallback to existing logic
            return self._generate_fallback_response(user_question, time_info)

    def _generate_fallback_response(self, user_question: str, time_info: dict) -> str:
        """Fallback response when AI fails"""
        current_day = time_info['current_day']
        hour = time_info['hour']
        is_open_day = current_day in ['Monday', 'Tuesday', 'Thursday']
        is_office_hours = 7 <= hour < 18
        is_open = is_open_day and is_office_hours
        
        if is_open:
            return (
                "I'm Dr. Tomar's virtual assistant. Our office is currently open until 6 PM. "
                "For specific scheduling questions, please call (425) 775-5162 and our team will assist you."
            )
        else:
            next_open = self.get_next_open_day(current_day)
            return (
                f"I'm Dr. Tomar's virtual assistant. Our office is currently closed. "
                f"We'll reopen on {next_open} from 7 AM to 6 PM. "
                "Please call (425) 775-5162 for scheduling assistance."
            )

    async def process_scheduling_query_async(
        self, 
        user_question: str, 
        context: str = "", 
        detected_lang: str = "English"
    ) -> AgentResponse:
        """Async version of scheduling query processing"""
        
        # If language not detected, detect it here
        if detected_lang == "English" and self.async_client:
            try:
                lang_response = await self.async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Detect primary language. For mixed languages like 'ky tm silver fillings krte ho', reply with the dominant language (Hindi, English, etc.). Reply only language name."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                detected_lang = lang_response.choices[0].message.content.strip()
                logger.info(f"⚡ Scheduling Agent Language Detection: {detected_lang}")
            except Exception as e:
                logger.warning(f"Language detection in scheduling agent failed: {e}")
                detected_lang = "English"
        
        return await self.process_scheduling_query(
            user_question=user_question,
            context=context,
            detected_lang=detected_lang  # FORCE pass language
        )
    
    async def process_scheduling_query(self, user_question: str, context: str = "", detected_lang: str = "English") -> AgentResponse:
        """Process scheduling queries with intelligent responses"""
        
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
        
        # Generate intelligent response based on context with language enforcement
        content = await self.generate_intelligent_response(user_question, intent, time_info, detected_lang)
        
        return AgentResponse(
            content=content,
            confidence=0.95,
            agent_type=AgentType.SCHEDULING,
            reasoning_steps=[f"Intent: {intent}", f"Day: {time_info['current_day']}", f"Language: {detected_lang}"],
            quality_score=95.0,
            attempts_used=1
        )

class MultiAgentOrchestrator:
    """Orchestrates multiple specialist agents for optimal responses"""
    
    def __init__(self, openai_client, pinecone_api_key: str):
        self.client = openai_client
        self.async_client = AsyncOpenAI(api_key=openai_client.api_key)
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
    
    def normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching"""
        import re
        
        # Basic normalization
        normalized = query.lower().strip()
        
        # Remove extra spaces and punctuation
        normalized = re.sub(r'[?!.]+', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Simple plural to singular (basic cases)
        normalized = re.sub(r'\bimplants\b', 'implant', normalized)
        normalized = re.sub(r'\bteeth\b', 'tooth', normalized)
        normalized = re.sub(r'\bcavities\b', 'cavity', normalized)
        
        return normalized.strip()
        
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
    
    def is_out_of_context(self, user_question: str, context: str = "") -> bool:
        """Detect if question is completely out of dental context - OPTIMIZED: No additional RAG calls"""
        
        # FIXED: Use only pre-retrieved context (no duplicate Pinecone calls)
        if context and len(context.strip()) > 50:
            return False
        
        # Quick check for common out-of-context patterns
        q = user_question.lower().strip()
        time_patterns = ['what time is it', "what's the time", 'current time', 'time now']
        if any(pattern in q for pattern in time_patterns):
            return True  # Definitely out-of-context
        
        try:
            out_of_context_prompt = f"""
Analyze this question and determine if it's related to dental/oral health or completely unrelated.

Question: "{user_question}"

IMPORTANT: Only mark as OUT_OF_CONTEXT if the question is completely unrelated to dental practice, office, or oral health.

Dental/oral health topics include: teeth, gums, mouth, dental procedures, oral hygiene, dental appointments, dental office, dentist services, oral pain, dental treatments, dental insurance, dental locations, staff languages, office staff, dental team, clinic information, office policies, patient services, dental practice, another practice, second practice, multiple locations, other offices, practice locations, dental clinics.

Non-dental topics include: weather, asking for current time ("what time is it", "what's the time", "current time"), sports, politics, general health (not oral), cooking, technology, entertainment, etc.

Respond with only "DENTAL" if it's dental-related or "OUT_OF_CONTEXT" if it's completely unrelated to dental/oral health:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": out_of_context_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "OUT_OF_CONTEXT"
            
        except Exception as e:
            print(f"Error in out-of-context detection: {e}")
            # Fallback to simple keyword check
            q = user_question.lower()
            
            # First check if it's definitely dental-related
            dental_keywords = [
                'dr tomar', 'doctor tomar', 'dentist', 'dental', 'practice', 
                'office', 'clinic', 'location', 'staff', 'team', 'appointment',
                'tooth', 'teeth', 'gum', 'mouth', 'oral', 'treatment'
            ]
            if any(keyword in q for keyword in dental_keywords):
                return False  # Definitely dental-related
            
            # Only very specific non-dental questions
            non_dental_patterns = [
                'what time is it', 'what\'s the time', 'current time',
                'weather', 'temperature', 'rain', 'snow',
                'sports', 'football', 'basketball', 'soccer',
                'politics', 'election', 'president',
                'cooking', 'recipe', 'food preparation',
                'music', 'song', 'movie', 'film'
            ]
            return any(pattern in q for pattern in non_dental_patterns)
    

    
    async def route_query(self, user_question: str, context: str = "", query_type: QueryType = None, detected_lang: str = "English") -> AgentResponse:
        """Route query to appropriate specialist agent - OPTIMIZED: No duplicate RAG calls"""
        
        # Check for out-of-context questions first using already retrieved context
        if self.is_out_of_context(user_question, context):
            return AgentResponse(
                content="I am unable to answer that question. I'm a virtual assistant for Dr. Meenakshi Tomar and can only help with dental and oral health related questions. How can I assist you with your dental needs today?",
                confidence=1.0,
                agent_type=AgentType.GENERAL,
                reasoning_steps=["Out-of-context question detected"],
                quality_score=100.0,
                attempts_used=1
            )
        
        # Use pre-classified query type (no duplicate classification)
        if not query_type:
            query_type = QueryType.GENERAL
        
        # Route to appropriate agent - FIXED: Use already retrieved context
        agent = self.agents.get(query_type.value, self.agents[AgentType.GENERAL])
        
        # FIXED: Always use provided context (no duplicate RAG calls)
        return await agent.process_query(user_question, context, query_type, "", detected_lang)
    

    

    
    async def process_consultation_async(self, user_question: str, conversation_history: str = "") -> AgentResponse:
        """ULTRA FAST: RAG cache + Final answer cache"""
        logger.info(f"Processing consultation: {user_question[:50]}...")
        
        # STEP 1: Final Answer Cache Check
        import hashlib, json
        normalized_query = self.normalize_query(user_question)
        query_hash = hashlib.md5(normalized_query.encode()).hexdigest()
        
        try:
            redis_client = self.rag_pipeline.redis_client if hasattr(self.rag_pipeline, 'redis_client') else None
            if redis_client:
                cached_response = redis_client.get(f"final_answer:{query_hash}")
                if cached_response:
                    response_data = json.loads(cached_response)
                    logger.info("⚡ FINAL ANSWER CACHE HIT - 0ms")
                    return AgentResponse(
                        content=response_data["content"],
                        confidence=response_data["confidence"],
                        agent_type=AgentType(response_data["agent_type"]),
                        reasoning_steps=response_data["reasoning_steps"],
                        quality_score=response_data["quality_score"],
                        attempts_used=response_data["attempts_used"]
                    )
        except Exception as e:
            logger.debug(f"Final answer cache check failed: {e}")
            pass
        
        # Check for greeting messages
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if user_question.lower().strip() in greeting_words:
            return AgentResponse(
                content="Welcome to Edmonds Bay Dental! How can I help you today!",
                confidence=1.0,
                agent_type=AgentType.GENERAL,
                reasoning_steps=["Detected greeting message"],
                quality_score=100.0,
                attempts_used=1
            )
        
        import time
        start_time = time.time()
        
        # 🚀 STEP 1: Query Classification FIRST
        classify_start = time.time()
        query_type = await self.query_classifier.classify_query_async(user_question, self.async_client)
        classify_time = (time.time() - classify_start) * 1000
        logger.info(f"⚡ Classification: {classify_time:.0f}ms | Type: {query_type.value}")
        
        # 🚀 STEP 2: Route based on classification
        if query_type == QueryType.SCHEDULING:
            logger.info("🔀 SCHEDULING → Direct agent (no cache/context)")
            
            # Detect language for scheduling
            try:
                lang_response = await self.async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Detect primary language. For mixed languages like 'ky tm silver fillings krte ho', reply with the dominant language (Hindi, English, etc.). Reply only language name."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                detected_lang = lang_response.choices[0].message.content.strip()
                logger.info(f"⚡ Scheduling Language: {detected_lang}")
            except Exception as e:
                detected_lang = "English"
                logger.warning(f"Language detection failed: {e}")
            
            scheduling_agent = self.agents[AgentType.SCHEDULING]
            response = await scheduling_agent.process_scheduling_query_async(
                user_question=user_question,
                context="",  # No context needed
                detected_lang=detected_lang  # Pass detected language
            )
            logger.info("🚫 No caching for scheduling - real-time data")
            return response
        
        # 🚀 STEP 3: For NON-SCHEDULING → Check cache first
        try:
            redis_client = self.rag_pipeline.redis_client if hasattr(self.rag_pipeline, 'redis_client') else None
            if redis_client:
                import hashlib
                normalized_query = self.normalize_query(user_question)
                query_hash = hashlib.md5(normalized_query.encode()).hexdigest()
                cached_response = redis_client.get(f"complete_response:{query_hash}")
                if cached_response:
                    import json
                    response_data = json.loads(cached_response)
                    logger.info("⚡ CACHE HIT - 0ms (non-scheduling)")
                    return AgentResponse(
                        content=response_data["content"],
                        confidence=response_data["confidence"],
                        agent_type=AgentType(response_data["agent_type"]),
                        reasoning_steps=response_data["reasoning_steps"],
                        quality_score=response_data["quality_score"],
                        attempts_used=response_data["attempts_used"]
                    )
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        # 🚀 STEP 4: Cache miss → Check FAQ/Persona
        faq_match = self.rag_pipeline._check_faq_context(user_question)
        if faq_match:
            logger.info("⚡ FAQ HIT - Skipping RAG")
            # Detect language for FAQ responses
            try:
                lang_response = await self.async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Detect primary language. For mixed languages like 'dr tomar silver fillings krti hai ky', reply with the dominant language (Hindi, English, etc.). Reply only language name."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                detected_lang = lang_response.choices[0].message.content.strip()
                logger.info(f"⚡ FAQ Language Detection: {detected_lang}")
            except Exception as e:
                detected_lang = "English"
                logger.warning(f"FAQ language detection failed: {e}")
            
            general_agent = self.agents[AgentType.GENERAL]
            response = await general_agent.process_query(
                user_question, self.rag_pipeline.faq_context, QueryType.GENERAL, "", detected_lang
            )
        else:
            persona_match = self.rag_pipeline._check_persona_context(user_question)
            if persona_match:
                logger.info("⚡ PERSONA HIT - Skipping RAG")
                # Detect language for persona responses
                try:
                    lang_response = await self.async_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Detect primary language. For mixed languages like 'dr tomar silver fillings krti hai ky', reply with the dominant language (Hindi, English, etc.). Reply only language name."},
                            {"role": "user", "content": user_question}
                        ],
                        temperature=0.1,
                        max_tokens=10
                    )
                    detected_lang = lang_response.choices[0].message.content.strip()
                    logger.info(f"⚡ Persona Language Detection: {detected_lang}")
                except Exception as e:
                    detected_lang = "English"
                    logger.warning(f"Persona language detection failed: {e}")
                
                general_agent = self.agents[AgentType.GENERAL]
                response = await general_agent.process_query(
                    user_question, persona_match, QueryType.GENERAL, "", detected_lang
                )
            else:
                # 🚀 STEP 5: FAQ/Persona miss → RAG call
                logger.info("📄 FAQ/Persona MISS - Running RAG")
                
                # Parallel: Language + RAG
                async def language_task():
                    start = time.time()
                    response = await self.async_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Detect primary language. For mixed languages like 'ky tm silver fillings krte ho', reply with the dominant language (Hindi, English, etc.). Reply only language name."},
                            {"role": "user", "content": user_question}
                        ],
                        temperature=0.1,
                        max_tokens=10
                    )
                    logger.info(f"⚡ Language: {(time.time()-start)*1000:.0f}ms")
                    return response.choices[0].message.content.strip()
                
                async def rag_task():
                    start = time.time()
                    loop = asyncio.get_event_loop()
                    context, _ = await loop.run_in_executor(
                        None,
                        lambda: self.rag_pipeline.retrieve_and_rank(user_question)
                    )
                    rag_time = (time.time() - start) * 1000
                    logger.info(f"📄 RAG: {rag_time:.0f}ms")
                    return context
                
                # Run Language + RAG in parallel
                results = await asyncio.gather(language_task(), rag_task(), return_exceptions=True)
                detected_lang = results[0] if not isinstance(results[0], Exception) else "English"
                context = results[1] if not isinstance(results[1], Exception) else ""
                
                # Route to appropriate agent
                if query_type == QueryType.EMERGENCY:
                    logger.info(f"🔀 ROUTE: Emergency → Direct")
                    emergency_agent = self.agents[AgentType.EMERGENCY]
                    response = await emergency_agent.process_emergency_query_async(
                        user_question, "", detected_lang
                    )
                    logger.info(f"🌍 Emergency processed in language: {detected_lang}")
                else:
                    logger.info(f"🔀 ROUTE: {query_type.value} → Agent")
                    agent = self.agents.get(query_type.value, self.agents[AgentType.GENERAL])
                    response = await agent.process_query(
                        user_question, context, query_type, "", detected_lang
                    )
                    logger.info(f"🌍 {query_type.value} processed in language: {detected_lang}")

        
        total_time = time.time() - start_time
        logger.info(f"🚀 Total time: {total_time*1000:.0f}ms | Agent: {response.agent_type.value}")
        
        # Cache Final Answer (only for non-scheduling)
        if query_type != QueryType.SCHEDULING:
            try:
                redis_client = self.rag_pipeline.redis_client if hasattr(self.rag_pipeline, 'redis_client') else None
                if redis_client:
                    response_data = {
                        "content": response.content,
                        "confidence": response.confidence,
                        "agent_type": response.agent_type.value,
                        "reasoning_steps": response.reasoning_steps,
                        "quality_score": response.quality_score,
                        "attempts_used": response.attempts_used
                    }
                    redis_client.setex(f"final_answer:{query_hash}", 86400, json.dumps(response_data))
                    logger.info("💾 FINAL ANSWER CACHED for future queries")
            except Exception as e:
                logger.debug(f"Final answer cache save failed: {e}")
        
        return response
    
    async def process_consultation(self, user_question: str, conversation_history: str = "") -> AgentResponse:
        """Main async consultation processing - FastAPI compatible"""
        return await self.process_consultation_async(user_question, conversation_history)
    


# Example usage
if __name__ == "__main__":
    # This would be integrated into the main application
    pass



