"""
Advanced RAG Pipeline with OpenAI Embeddings
Implements sophisticated retrieval and ranking using OpenAI embeddings only
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
import asyncio
from functools import lru_cache
import hashlib
from pinecone import Pinecone
from loguru import logger
import tiktoken
import redis
import json
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Redis CLOUD - Static configuration
try:
    redis_client = redis.Redis(
        host='redis-11326.c258.us-east-1-4.ec2.cloud.redislabs.com',
        port=11326,
        username='default',
        password='vkvFfdBQ0c5y6mbXeYORL1bPdNu73FSX',
        db=0,
        decode_responses=False,
        socket_timeout=1,  # Faster timeout
        socket_connect_timeout=1
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("☁️ Redis connected: redis-11326.c258.us-east-1-4.ec2.cloud.redislabs.com:11326")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"⚠️ Redis failed - memory cache only. Error: {e}")
    redis_client = None

# ULTRA FAST embedding cache
embedding_cache = {}

def cached_embed(text: str, api_key: str):
    """SUPER FAST embedding cache - memory first, Redis backup"""
    cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    
    # SPEED HACK 1: Memory cache first (instant)
    if cache_key in embedding_cache:
        logger.info("⚡ Embedding cache HIT (memory) - 0ms")
        return embedding_cache[cache_key]
    
    # SPEED HACK 2: Redis backup
    if REDIS_AVAILABLE:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                embedding_data = json.loads(cached)
                embedding_cache[cache_key] = embedding_data["embedding"]  # Store in memory
                logger.info("📄 Embedding cache HIT (Redis) - 5ms")
                return embedding_data["embedding"]
        except:
            pass
    
    # Generate new embedding
    logger.info("🔄 Generating new embedding - 150ms")
    client = openai.OpenAI(
        api_key=api_key, 
        timeout=3.0,  # Aggressive timeout
        max_retries=0  # No retries for speed
    )
    embedding = client.embeddings.create(
        model="text-embedding-3-small",  # Faster model
        input=text[:500]  # Even smaller text limit
    ).data[0].embedding
    
    # SPEED HACK 3: Store in both caches
    embedding_cache[cache_key] = embedding
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(cache_key, 86400, json.dumps({"embedding": embedding}))
        except:
            pass
    
    return embedding

@dataclass
class RetrievedChunk:
    content: str
    score: float
    chunk_id: str
    metadata: Dict
    relevance_score: float = 0.0
    semantic_score: float = 0.0

class AdvancedRAGPipeline:
    """Enhanced RAG system using OpenAI embeddings for retrieval and ranking"""

    def __init__(self, openai_client, pinecone_api_key: str, index_name: str = "rag-index"):
        self.openai_client = openai_client
        self.async_client = AsyncOpenAI(api_key=openai_client.api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        # Add redis client reference for caching
        self.redis_client = redis_client if REDIS_AVAILABLE else None

        # Token counter for context management
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.max_context_tokens = 3000
        
        # Optimized chunk settings
        self.chunk_size = 500  # chars
        self.chunk_overlap = 50  # chars
        
        # FAQ context
        self.faq_context = self._get_faq_context()

        logger.info("Advanced RAG Pipeline initialized with OpenAI embeddings and FAQ context")
    
    def enhanced_retrieval(
        self,
        query: str,
        top_k: int = 1
    ) -> List[RetrievedChunk]:
        """
        Enhanced retrieval using OpenAI embeddings with query expansion
        """

        # 1. Primary vector-based retrieval using OpenAI embeddings
        primary_chunks = self._vector_retrieval(query, top_k)

        # 2. Query expansion for better coverage - DISABLED for speed optimization
        # expanded_queries = self.query_expansion(query)
        expanded_queries = [query]  # Use only original query for speed
        if not isinstance(expanded_queries, list):
            expanded_queries = [query]
        all_chunks = primary_chunks.copy()

        # 3. Retrieve for expanded queries
        if len(expanded_queries) > 1:
            for expanded_query in expanded_queries[1:]:  # Skip original query
                expanded_chunks = self._vector_retrieval(expanded_query, max(1, top_k // 2))
                all_chunks.extend(expanded_chunks)

        # 4. Remove duplicates and re-rank
        final_chunks = self._deduplicate_and_rank(all_chunks, top_k)

        logger.info(f"Retrieved {len(final_chunks)} chunks using enhanced OpenAI approach")
        return final_chunks
    
    async def _get_embedding_cached_async(self, query: str):
        """Async cached embedding generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cached_embed, query, self.openai_client.api_key)
    
    async def _vector_retrieval_async(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """LIGHTNING FAST RAG - under 500ms target"""
        
        query_hash = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        
        # SPEED HACK 1: Memory cache first (instant)
        cache_key = f"rag_result:{query_hash}"
        if cache_key in embedding_cache:
            logger.info("⚡ RAG cache HIT (memory) - 0ms")
            return embedding_cache[cache_key]
        
        # SPEED HACK 2: Redis cache (5ms)
        if REDIS_AVAILABLE:
            try:
                cached_results = redis_client.get(f"retrieval:{query_hash}")
                if cached_results:
                    chunks = pickle.loads(cached_results)
                    embedding_cache[cache_key] = chunks  # Store in memory
                    logger.info("📄 RAG cache HIT (Redis) - 5ms")
                    return chunks
            except:
                pass
        
        try:
            # SPEED HACK 3: Direct sync embedding (faster than async overhead)
            query_embedding = cached_embed(query, self.openai_client.api_key)
            
            # SPEED HACK 4: Direct Pinecone query (no async overhead)
            results = self.index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True,
                include_values=False
            )
            
            # SPEED HACK 5: Minimal chunk creation
            chunks = [
                RetrievedChunk(
                    content=match['metadata'].get('text', ''),
                    score=match['score'],
                    chunk_id=match['id'],
                    metadata=match['metadata']
                ) for match in results['matches']
            ]
            
            # SPEED HACK 6: Store in both caches
            embedding_cache[cache_key] = chunks
            if REDIS_AVAILABLE:
                try:
                    redis_client.setex(f"retrieval:{query_hash}", 86400, pickle.dumps(chunks))
                except:
                    pass
            
            return chunks
            
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return []
    
    def _vector_retrieval(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Redis cached vector-based retrieval using OpenAI embeddings"""
        import time
        total_start = time.time()
        
        query_hash = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        
        # Try Redis cache first
        if REDIS_AVAILABLE:
            try:
                redis_start = time.time()
                cached_results = redis_client.get(f"retrieval:{query_hash}")
                redis_time = (time.time() - redis_start) * 1000
                if cached_results:
                    logger.info(f"📄 RAG cache HIT - {redis_time:.0f}ms retrieval")
                    return pickle.loads(cached_results)
                logger.info(f"📄 Redis check: {redis_time:.0f}ms (MISS)")
            except Exception as e:
                logger.warning(f"Redis retrieval cache error: {e}")
        
        try:
            # Get cached embedding
            embed_start = time.time()
            query_embedding = cached_embed(query, self.openai_client.api_key)
            embed_time = (time.time() - embed_start) * 1000
            logger.info(f"🔄 Embedding: {embed_time:.0f}ms")
            
            # Search in Pinecone (sync version)
            pinecone_start = time.time()
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            pinecone_time = (time.time() - pinecone_start) * 1000
            logger.info(f"🌲 Pinecone: {pinecone_time:.0f}ms")
            
            chunks = []
            for match in results['matches']:
                chunk = RetrievedChunk(
                    content=match['metadata'].get('text', ''),
                    score=match['score'],
                    chunk_id=match['id'],
                    metadata=match['metadata']
                )
                chunks.append(chunk)
            
            # Cache results in Redis
            total_time = (time.time() - total_start) * 1000
            logger.info(f"🚀 Total RAG: {total_time:.0f}ms")
            
            if REDIS_AVAILABLE:
                try:
                    redis_client.setex(f"retrieval:{query_hash}", 86400, pickle.dumps(chunks))
                    logger.info("📄 RAG cache MISS - cached for next time")
                except Exception as e:
                    logger.warning(f"Redis retrieval cache save error: {e}")
            else:
                logger.info("📄 RAG cache MISS - cached for next time")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return []
    
    def _deduplicate_and_rank(
        self,
        chunks: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        """Remove duplicates and rank chunks by relevance score"""

        # Remove duplicates based on chunk_id
        unique_chunks = {}
        for chunk in chunks:
            if chunk.chunk_id not in unique_chunks:
                unique_chunks[chunk.chunk_id] = chunk
            else:
                # Keep the one with higher score
                if chunk.score > unique_chunks[chunk.chunk_id].score:
                    unique_chunks[chunk.chunk_id] = chunk

        # Take first top_k chunks (no slow sorting)
        final_chunks = list(unique_chunks.values())[:top_k]

        # Update relevance scores
        for chunk in final_chunks:
            chunk.relevance_score = chunk.score
            chunk.semantic_score = chunk.score

        return final_chunks
    

    
    def retrieve_and_rank(self, query: str) -> Tuple[str, List[RetrievedChunk]]:
        """Smart retrieval: FAQ first, then RAG if needed"""
        import time
        start_time = time.time()
        
        # 1️⃣ Check FAQ context first (instant)
        faq_match = self._check_faq_context(query)
        if faq_match:
            logger.info("⚡ FAQ context HIT - skipping RAG retrieval")
            return self.faq_context, []
        
        # 2️⃣ Check persona/prompt context (fast)
        persona_match = self._check_persona_context(query)
        if persona_match:
            logger.info("⚡ Persona context HIT - skipping RAG retrieval")
            return persona_match, []
        
        # 3️⃣ No context found → run RAG retrieval
        logger.info("📄 No FAQ/persona context - running RAG retrieval")
        chunks = self.enhanced_retrieval(query)
        context = self.context_optimization(chunks, query)
        
        end_time = time.time()
        logger.info(f"📄 RAG retrieval time: {(end_time - start_time)*1000:.0f}ms")
        
        return context, chunks
    
    async def enhanced_retrieval_async(self, query: str, top_k: int = 1) -> List[RetrievedChunk]:
        """LIGHTNING FAST retrieval - under 500ms"""
        # SPEED HACK: Direct sync call (faster than async overhead)
        chunks = self._vector_retrieval(query, 1)
        logger.info(f"Retrieved {len(chunks)} chunks (LIGHTNING FAST)")
        return chunks
    
    async def retrieve_and_rank_async(self, query: str) -> Tuple[str, List[RetrievedChunk]]:
        """LIGHTNING SPEED: under 500ms target"""
        import time
        start_time = time.time()
        
        # SPEED HACK: Direct sync call (no async overhead)
        chunks = self._vector_retrieval(query, 1)
        
        # SPEED HACK: Minimal context building
        if chunks and chunks[0].content:
            context = f"{self.faq_context}\n\n{chunks[0].content[:400]}"  # Smaller limit
        else:
            context = self.faq_context
        
        end_time = time.time()
        logger.info(f"📄 RAG time (LIGHTNING): {(end_time - start_time)*1000:.0f}ms")
        
        return context, chunks
    
    def _calculate_relevance_boost(self, chunk: RetrievedChunk, query: str) -> float:
        """Calculate relevance boost based on content analysis"""

        content_lower = chunk.content.lower()
        query_lower = query.lower()

        # Boost for exact keyword matches
        query_words = query_lower.split()
        exact_matches = sum(1 for word in query_words if word in content_lower)
        exact_match_boost = exact_matches * 0.1

        # Boost for medical/dental terms
        medical_terms = ['tooth', 'dental', 'pain', 'treatment', 'procedure', 'diagnosis']
        medical_boost = sum(0.05 for term in medical_terms if term in content_lower)

        # Boost for Dr. Meenakshi specific content
        doctor_terms = ['meenakshi', 'tomar', 'edmonds', 'laser', 'wcli']
        doctor_boost = sum(0.1 for term in doctor_terms if term in content_lower)

        total_boost = exact_match_boost + medical_boost + doctor_boost
        return min(total_boost, 0.5)  # Cap at 0.5
    
    def _check_faq_context(self, query: str) -> str:
        """Check if query matches FAQ patterns (instant)"""
        q = query.lower()
        
        # FAQ keyword matching
        faq_patterns = {
            'appointment': 'schedule an appointment',
            'bitcoin': 'accept Bitcoin',
            'laughing gas': 'offer laughing gas', 
            'silver filling': 'offer silver fillings',
            'botox': 'offer Botox',
            'toddler': 'treat toddlers',
            'hygienist': 'have a hygienist',
            'medicare': 'accept Medicare',
            'apple health': 'accept Apple Health',
            'coffee': 'offer free coffee',
            'bathroom': 'use the bathroom',
            'location': 'another dental location',
            'time': 'What time is it'
        }
        
        for keyword, pattern in faq_patterns.items():
            if keyword in q:
                return self.faq_context
        
        return None
    
    def _check_persona_context(self, query: str) -> str:
        """Check if query matches persona/prompt patterns"""
        q = query.lower()
        
        # Common dental questions that don't need RAG
        persona_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon',
            'how are you', 'what do you do', 'who are you',
            'office hours', 'phone number', 'address', 'location'
        ]
        
        if any(pattern in q for pattern in persona_patterns):
            return "I'm Dr. Meenakshi Tomar's virtual assistant. I can help with dental questions, appointments, and office information. How can I assist you today?"
        
        return None
    
    def _get_faq_context(self) -> str:
        """Get FAQ context with template variables replaced"""
        return """
FREQUENTLY ASKED QUESTIONS (FAQ):



Q. Can you schedule an appointment for me? 
A. Unfortunately, I cannot book an appointment directly. Please call our clinic at (425) 775-5162 and our team will be happy to help you with scheduling.




Q. Do you accept Bitcoin? 
A. Edmonds Bay Dental does not accept Bitcoin. We accept cash, all major credit cards, and insurance.

Q. Do you offer laughing gas? 
A. Dr. Tomar does not use laughing gas. Instead, she uses single anesthesia to ensure comfort during procedures.

Q. Do you offer silver fillings? 
A. Dr. Tomar does not provide silver (amalgam) fillings. She prefers composite fillings, which are more aesthetically pleasing and often lead to better outcomes.

Q. Do you offer Botox? 
A. Dr. Tomar does not perform Botox treatments. Please contact our clinic for details on the procedures we do offer.

Q. Do you treat toddlers? 
A. Yes, Dr. Tomar treats toddlers. Please contact our clinic for specific details and to schedule an appointment.

Q. Do you have a hygienist? 
A. Edmonds Bay Dental does not have a hygienist. Instead,we have  dental assistants recommended by Dr. Tomar are highly qualified and dedicated to providing exceptional patient care.

Q. Do you accept Medicare? 
A. Edmonds Bay Dental does not participate in Medicare insurance plans. Please contact our clinic to learn about our in-house plans and options.

Q. Do you accept Apple Health? 
A. Edmonds Bay Dental does not participate in Apple Health plans. Please contact our clinic to learn about our in-house plans and options.

Q. Do you offer free coffee at your clinic? 
A. We do not offer free coffee, but we do our best to make your visit as comfortable as possible.

Q. Can I use the bathroom at your clinic? 
A. Our facilities are reserved for patients with appointments.

Q. Do you have another dental location? 
A. Yes. Dr. Tomar also practices at Pacific Highway Dental Clinic.Location -27020 Pacific Highway South, Suite C,

Kent, WA 98032. Please contact the clinic for exact address and hours.

Q. What time is it right now?
A. I am unable to answer that question. I'm a virtual assistant for Dr. Meenakshi Tomar and can only assist with dental and oral health-related inquiries. How can I help you with your dental needs today?
"""
    
    def context_optimization(self, chunks: List[RetrievedChunk], query: str) -> str:
        """Optimize context for token limits and relevance with FAQ integration"""
        
        context_parts = []
        total_tokens = 0
        
        # Add FAQ context first
        faq_tokens = len(self.encoding.encode(self.faq_context))
        if faq_tokens < self.max_context_tokens // 3:  # Use max 1/3 for FAQ
            context_parts.append(self.faq_context + "\n\n")
            total_tokens += faq_tokens
        
        if not chunks:
            return self.faq_context
        
        # Use first 2 chunks only (no slow sorting)
        sorted_chunks = chunks[:2]
        
        # Add query-specific context header
        context_header = f"RELEVANT MEDICAL KNOWLEDGE FOR: {query}\n\n"
        header_tokens = len(self.encoding.encode(context_header))
        total_tokens += header_tokens
        
        # Add knowledge base chunks ONLY ONCE
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"CONTEXT {i+1}:\n{chunk.content}\n\n"
            chunk_tokens = len(self.encoding.encode(chunk_text))
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        final_context = "".join(context_parts)
        
        logger.info(f"Optimized context with FAQ: {total_tokens} tokens, {len(sorted_chunks) if chunks else 0} chunks")
        return final_context
    
    def query_expansion(self, original_query: str) -> List[str]:
        """Expand query with related terms for better retrieval"""
        
        expansion_prompt = f"""Given this dental/medical query, generate 3-5 related search terms or phrases that would help find relevant information:

Original query: "{original_query}"

Generate related terms that cover:
1. Medical/dental terminology
2. Symptoms or conditions mentioned
3. Treatment options
4. Related procedures or concepts

Return only the related terms, one per line:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            expanded_terms = response.choices[0].message.content.strip().split('\n')
            expanded_terms = [term.strip() for term in expanded_terms if term.strip()]
            
            logger.info(f"Query expanded with {len(expanded_terms)} additional terms")
            return [original_query] + expanded_terms
            
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return [original_query]
    


# Example usage and testing
if __name__ == "__main__":
    # This would be used in the main application
    pass
