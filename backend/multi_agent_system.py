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
- Monday: 7:00 AM - 6:00 PM (OPEN)
- Tuesday: 7:00 AM - 6:00 PM (OPEN) 
- Wednesday: CLOSED
- Thursday: 7:00 AM - 6:00 PM (OPEN)
- Friday: CLOSED
- Saturday: CLOSED
- Sunday: CLOSED

LOCATION: Edmonds, Washington (Pacific Time Zone - America/Los_Angeles)
PHONE: (425) 775-5162
""",
            AgentType.GENERAL: """
GENERAL CONSULTATION SPECIALIZATION:
- Dr. Tomar is expert in comprehensive dental care coordination
- Dr. Tomar is skilled in patient communication and education
- Dr. Tomar is experienced in holistic oral health assessment
- Dr. Tomar focuses on overall patient wellbeing and care continuity
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
            classifier = QueryClassifier()
            agent_query_type = classifier.classify_query(user_question)
        
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
    
    def check_office_status(self, day: str) -> Dict[str, any]:
        """Check if office is open on given day"""
        open_days = {'Monday', 'Tuesday', 'Thursday'}
        is_open = day in open_days
        
        return {
            'is_open': is_open,
            'hours': '7:00 AM - 6:00 PM' if is_open else 'Closed',
            'day': day
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
    
    def detect_scheduling_intent(self, user_question: str) -> str:
        """Dynamically detect scheduling intent using AI"""
        
        intent_prompt = f"""
Analyze this user question and classify the scheduling intent. Return ONLY the intent category:

User Question: "{user_question}"

Intent Categories:
- schedule_request: User wants to book/schedule/make an appointment
- same_day_request: User asking about same-day appointments or availability today
- see_me_request: User asking "can you see me" or when they can be seen
- hours_inquiry: User asking about office hours, opening times, or when clinic is open
- modify_appointment: User wants to cancel, reschedule, or change existing appointment
- cost_inquiry: User asking about prices, costs, or fees for procedures
- insurance_inquiry: User asking about insurance acceptance or coverage
- general_scheduling: Any other scheduling-related question

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
                           "hours_inquiry", "modify_appointment", "cost_inquiry", 
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
User wants to schedule appointment. Check current day status first:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

If TODAY IS OPEN ({time_info['current_day']} and office is OPEN): 
Our clinic is open at the moment, so please give us a call, and we can try to make an appointment for a time that works for your schedule.

**Contact Information:**
• Phone: (425) 775-5162
• Location: Edmonds Bay Dental, Edmonds, WA
• Available: We can find a time that works for your schedule

If TODAY IS CLOSED ({time_info['current_day']} and office is CLOSED):
While I am unable to make or modify appointments, our scheduling team is available to help.

**Scheduling Team:**
• Phone: (425) 775-5162
• Available: 7 AM to 6 PM, Mon, Tue, and Thu
• Service: They will be happy to assist you

CRITICAL FORMATTING REQUIREMENTS:
1. Check the CURRENT DAY STATUS above to determine if office is open or closed TODAY
2. MANDATORY: Each bullet point MUST be on a separate line with line breaks
3. MANDATORY: Format exactly like the examples above with proper spacing
4. Use the correct response based on whether office is OPEN or CLOSED today

User: "{user_question}"
""",
            
            "same_day_request": f"""
{context}
User asking about same-day appointments. Check current day status first:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

Edmonds Bay Dental does offer same-day appointments when possible.

**Availability Details:**
• Today: {time_info['current_day']} - {'Available (Office Open)' if current_day_status['is_open'] else 'Not Available (Office Closed)'}
• Contact: (425) 775-5162 to check specific availability
• Quick Response: Call now for fastest scheduling

IMPORTANT: 
1. Check the CURRENT DAY STATUS above to determine availability
2. Each bullet point must be on a separate line with proper spacing
3. Show correct availability based on office status

User: "{user_question}"
""",
            
            "see_me_request": f"""
{context}
User asking "can you see me". Check current day status first:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

If TODAY IS OPEN ({time_info['current_day']} and office is OPEN):
Dr. Tomar sees patients till 6 PM today ({time_info['current_day']}).

**Today's Details:**
• Status: Open until 6 PM
• Contact: (425) 775-5162 to check specific availability
• Location: Edmonds Bay Dental, Edmonds, WA

If TODAY IS CLOSED ({time_info['current_day']} and office is CLOSED):
Dr. Tomar's office is closed today.

**Next Available:**
• Days: Monday, Tuesday, and Thursday
• Hours: 7 AM to 6 PM
• Contact: (425) 775-5162 to schedule

IMPORTANT: 
1. Check the CURRENT DAY STATUS above to determine if office is open or closed TODAY
2. Each bullet point must be on a separate line with proper spacing
3. Use the correct response based on whether office is OPEN or CLOSED today

User: "{user_question}"
""",
            
            "hours_inquiry": f"""
{context}
User asking about office hours. Check current day status and respond accordingly:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

If TODAY IS OPEN ({time_info['current_day']} and office is OPEN):
Edmonds Bay Dental is open today until 6 PM.

Office Hours:
• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Wednesday: CLOSED
• Thursday: 7 AM - 6 PM
• Friday-Sunday: CLOSED

Contact: (425) 775-5162 for appointments

If TODAY IS CLOSED ({time_info['current_day']} and office is CLOSED):
Edmonds Bay Dental is closed today.

Office Hours:
• Monday: 7 AM - 6 PM
• Tuesday: 7 AM - 6 PM
• Thursday: 7 AM - 6 PM
• Wednesday, Friday-Sunday: CLOSED

Contact: (425) 775-5162 to schedule

IMPORTANT: 
1. Check the CURRENT DAY STATUS above to determine if office is open or closed TODAY
2. Each bullet point must be on a separate line with proper spacing
3. Use the correct response based on whether office is OPEN or CLOSED today

User: "{user_question}"
""",
            
            "modify_appointment": f"""
{context}
User wants to cancel/reschedule. Check current day status first:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

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
User asking about costs. Check current day status first:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

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
User asking about insurance. Check current day status first:

CURRENT DAY STATUS: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}

Edmonds Bay Dental participates in most Private Dental PPO plans.

**Insurance Details:**
• Accepted: Most Private Dental PPO plans
• Verification: Call (425) 775-5162 to confirm your specific plan
• Benefits: Our team can help maximize your coverage

IMPORTANT: 
1. Check the CURRENT DAY STATUS above for reference
2. Each bullet point must be on a separate line with proper spacing
3. Always provide insurance information regardless of current status

User: "{user_question}"
"""
        }
        
        return intent_responses.get(intent, f"{context}\nGeneral scheduling inquiry. Provide helpful information and direct to (425) 775-5162.\nUser: {user_question}")
    
    def process_scheduling_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Process scheduling-specific queries with intent-based responses"""
        
        # Get current time info
        time_info = self.get_current_time_info()
        current_day_status = self.check_office_status(time_info['current_day'])
        tomorrow_day_status = self.check_office_status(time_info['tomorrow_day'])
        
        # Debug logging
        logger.info(f"🗓️ Current Day: {time_info['current_day']} - {'OPEN' if current_day_status['is_open'] else 'CLOSED'}")
        logger.info(f"🗓️ Tomorrow: {time_info['tomorrow_day']} - {'OPEN' if tomorrow_day_status['is_open'] else 'CLOSED'}")
        logger.info(f"🕐 Current Time: {time_info['current_time']} ({time_info['timezone']})")
        
        # Detect specific intent
        intent = self.detect_scheduling_intent(user_question)
        logger.info(f"🎯 Detected Intent: {intent}")
        
        # Generate intent-specific prompt
        scheduling_prompt = self._get_intent_prompt(intent, time_info, current_day_status, tomorrow_day_status, user_question)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": scheduling_prompt}],
                temperature=0.1,
                max_tokens=300
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
        self.query_classifier = QueryClassifier()
        
        # Initialize specialist agents
        self.agents = {
            AgentType.DIAGNOSTIC.value: BaseAgent(openai_client, AgentType.DIAGNOSTIC),
            AgentType.TREATMENT.value: BaseAgent(openai_client, AgentType.TREATMENT),
            AgentType.PREVENTION.value: BaseAgent(openai_client, AgentType.PREVENTION),
            AgentType.EMERGENCY.value: BaseAgent(openai_client, AgentType.EMERGENCY),
            AgentType.SCHEDULING: SchedulingAgent(openai_client),
            AgentType.GENERAL.value: BaseAgent(openai_client, AgentType.GENERAL)
        }
        
        # Monitoring disabled for compatibility
        logger.info("Multi-Agent System monitoring via standard logging")
        
        logger.info("Multi-Agent System initialized with all specialist agents")
    
    def route_query(self, user_question: str, context: str = "") -> AgentResponse:
        """Route query to appropriate specialist agent"""
        
        # Classify the query
        query_type = self.query_classifier.classify_query(user_question)
        
        # Special handling for scheduling queries
        if query_type == QueryType.SCHEDULING or self._is_scheduling_query(user_question):
            return self.agents[AgentType.SCHEDULING].process_scheduling_query(user_question, context)
        
        # Route to appropriate agent
        agent = self.agents.get(query_type.value, self.agents[AgentType.GENERAL.value])
        
        # Get relevant context from RAG if needed
        if context:
            rag_context = context
        else:
            rag_context = self.rag_pipeline.get_relevant_context(user_question)
        
        return agent.process_query(user_question, rag_context, query_type)
    
    def _is_scheduling_query(self, user_question: str) -> bool:
        """AI-powered scheduling query detection"""
        
        try:
            intent_prompt = f"""
Analyze this user question and determine if it's related to scheduling/appointments.

User Question: "{user_question}"

Is this question about:
- Scheduling appointments
- Booking appointments  
- Office hours/availability
- Asking to see the doctor
- Appointment changes/cancellations
- Same-day appointments
- Insurance/cost related to scheduling

Respond with only: YES or NO
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            logger.error(f"AI scheduling detection error: {e}")
            # Fallback to basic keyword check
            return "appointment" in user_question.lower() or "schedule" in user_question.lower()
    
    def process_consultation(self, user_question: str, conversation_history: str = "") -> AgentResponse:
        """Main consultation processing with agent orchestration"""
        
        logger.info(f"Processing consultation: {user_question[:50]}...")
        
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
        """Add consultation recommendation for disease/condition related queries"""
        
        # Keywords that indicate disease, condition, or health concerns
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
        has_health_concern = any(keyword in question_lower for keyword in disease_keywords)
        
        # Check if consultation info is already in response
        has_consultation_info = "(425) 775-5162" in response_content or "Dr. Meenakshi Tomar" in response_content
        
        # Add consultation recommendation if needed
        if has_health_concern and not has_consultation_info:
            consultation_text = "\n\nFor proper diagnosis and personalized treatment, I recommend consulting with Dr. Meenakshi Tomar directly. You can reach us at (425) 775-5162 to schedule an appointment."
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
            secondary_response = self.agents[AgentType.GENERAL.value].process_query(user_question, context)
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
