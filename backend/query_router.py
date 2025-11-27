"""
Query Router System
Implements: 1) Classifier → 2) If scheduling → skip RAG → scheduling agent → return
           3) Else → call RAG async → optimized context → correct agent
"""

import asyncio
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from advanced_prompts import QueryClassifier, QueryType
from multi_agent_system import MultiAgentOrchestrator, AgentResponse, AgentType
from advanced_rag import AdvancedRAGPipeline

class RouteDecision(Enum):
    SCHEDULING_DIRECT = "scheduling_direct"
    RAG_THEN_AGENT = "rag_then_agent"

@dataclass
class RoutingResult:
    decision: RouteDecision
    query_type: QueryType
    agent_response: Optional[AgentResponse] = None
    context: str = ""

class QueryRouter:
    """Fast query routing system with optimized paths"""
    
    def __init__(self, openai_client, pinecone_api_key: str):
        self.client = openai_client
        self.classifier = QueryClassifier(openai_client)
        self.orchestrator = MultiAgentOrchestrator(openai_client, pinecone_api_key)
        self.rag_pipeline = AdvancedRAGPipeline(openai_client, pinecone_api_key)
        
    async def route_query(self, user_question: str) -> AgentResponse:
        """
        Main routing logic:
        1) Run classifier
        2) If scheduling → skip RAG → call scheduling agent → return
        3) Else → call RAG async → optimized context → correct agent
        """
        
        # Step 1: Run classifier
        query_type = await self.classifier.classify_query_async(user_question, self.client)
        
        # Step 2: If scheduling → skip RAG → direct to scheduling agent
        if query_type == QueryType.SCHEDULING:
            logger.info("🔀 ROUTE: Scheduling detected → Direct to scheduling agent (no RAG)")
            scheduling_agent = self.orchestrator.agents[AgentType.SCHEDULING]
            return await scheduling_agent.process_scheduling_query_async(user_question)
        
        # Step 3: Else → RAG async → optimized context → correct agent
        logger.info(f"🔀 ROUTE: {query_type.value} detected → RAG + Agent pipeline")
        
        # Get optimized context from RAG
        context, chunks = await self.rag_pipeline.retrieve_and_rank_async(user_question)
        
        # Route to correct agent with context
        return await self.orchestrator.route_query(user_question, context, query_type)

    async def process_with_routing(self, user_question: str) -> AgentResponse:
        """Process query with optimized routing"""
        import time
        start_time = time.time()
        
        try:
            response = await self.route_query(user_question)
            
            end_time = time.time()
            logger.info(f"🚀 Total routing time: {(end_time - start_time)*1000:.0f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Fallback to general agent
            general_agent = self.orchestrator.agents[AgentType.GENERAL]
            return await general_agent.process_query(
                user_question, 
                "", 
                QueryType.GENERAL
            )