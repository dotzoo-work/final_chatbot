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

class MultiAgentOrchestrator:
    """Orchestrates multiple specialist agents for optimal responses"""
    
    def __init__(self, openai_client, pinecone_api_key: str):
        self.client = openai_client
        self.rag_pipeline = AdvancedRAGPipeline(openai_client, pinecone_api_key)
        self.query_classifier = QueryClassifier()
        
        # Initialize specialist agents
        self.agents = {
            AgentType.DIAGNOSTIC: BaseAgent(openai_client, AgentType.DIAGNOSTIC),
            AgentType.TREATMENT: BaseAgent(openai_client, AgentType.TREATMENT),
            AgentType.PREVENTION: BaseAgent(openai_client, AgentType.PREVENTION),
            AgentType.EMERGENCY: BaseAgent(openai_client, AgentType.EMERGENCY),
            AgentType.GENERAL: BaseAgent(openai_client, AgentType.GENERAL)
        }
        
        # Monitoring disabled for compatibility
        logger.info("Multi-Agent System monitoring via standard logging")
        
        logger.info("Multi-Agent System initialized with all specialist agents")
    
    def route_query(self, user_question: str) -> AgentType:
        """Route query to appropriate specialist agent"""
        query_type = self.query_classifier.classify_query(user_question)
        
        # Map QueryType to AgentType
        routing_map = {
            QueryType.DIAGNOSIS: AgentType.DIAGNOSTIC,
            QueryType.TREATMENT: AgentType.TREATMENT,
            QueryType.PREVENTION: AgentType.PREVENTION,
            QueryType.EMERGENCY: AgentType.EMERGENCY,
            QueryType.PROCEDURE: AgentType.TREATMENT,
            QueryType.COST: AgentType.GENERAL,
            QueryType.GENERAL: AgentType.GENERAL
        }
        
        return routing_map.get(query_type, AgentType.GENERAL)
    
    def process_consultation(self, user_question: str, conversation_history: str = "") -> AgentResponse:
        """Main consultation processing with agent orchestration"""
        
        logger.info(f"Processing consultation: {user_question[:50]}...")
        
        # 1. Retrieve relevant context
        context, retrieved_chunks = self.rag_pipeline.retrieve_and_rank(user_question)
        
        # 2. Route to appropriate agent
        selected_agent_type = self.route_query(user_question)
        selected_agent = self.agents[selected_agent_type]
        
        logger.info(f"Routed to {selected_agent_type.value} agent")
        
        # 3. Process with specialist agent
        response = selected_agent.process_query(user_question, context, conversation_history=conversation_history)
        
        # 4. Add consultation recommendation for disease/condition queries
        response.content = self._add_consultation_recommendation(response.content, user_question)
        
        # 5. Log metrics for monitoring (simplified logging)
        logger.info(f"Consultation metrics - Agent: {selected_agent_type.value}, "
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
        primary_agent_type = self.route_query(user_question)
        primary_response = self.agents[primary_agent_type].process_query(user_question, context)
        agent_responses.append(primary_response)
        
        # Secondary agent (general consultation for broader perspective)
        if primary_agent_type != AgentType.GENERAL:
            secondary_response = self.agents[AgentType.GENERAL].process_query(user_question, context)
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
