"""
Advanced RAG Pipeline with OpenAI Embeddings
Implements sophisticated retrieval and ranking using OpenAI embeddings only
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openai
from pinecone import Pinecone
from loguru import logger
import tiktoken

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
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

        # Token counter for context management
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.max_context_tokens = 3000

        logger.info("Advanced RAG Pipeline initialized with OpenAI embeddings")
    
    def enhanced_retrieval(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievedChunk]:
        """
        Enhanced retrieval using OpenAI embeddings with query expansion
        """

        # 1. Primary vector-based retrieval using OpenAI embeddings
        primary_chunks = self._vector_retrieval(query, top_k)

        # 2. Query expansion for better coverage
        expanded_queries = self.query_expansion(query)
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
    
    def _vector_retrieval(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Traditional vector-based retrieval using OpenAI embeddings"""
        
        try:
            # Get query embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            chunks = []
            for match in results['matches']:
                chunk = RetrievedChunk(
                    content=match['metadata'].get('text', ''),
                    score=match['score'],
                    chunk_id=match['id'],
                    metadata=match['metadata']
                )
                chunks.append(chunk)
            
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

        # Sort by score and return top_k
        final_chunks = sorted(
            unique_chunks.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]

        # Update relevance scores
        for chunk in final_chunks:
            chunk.relevance_score = chunk.score
            chunk.semantic_score = chunk.score

        return final_chunks
    
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
    
    def context_optimization(self, chunks: List[RetrievedChunk], query: str) -> str:
        """Optimize context for token limits and relevance"""
        
        if not chunks:
            return ""
        
        # Sort chunks by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)
        
        context_parts = []
        total_tokens = 0
        
        # Add query-specific context header
        context_header = f"RELEVANT MEDICAL KNOWLEDGE FOR: {query}\n\n"
        header_tokens = len(self.encoding.encode(context_header))
        total_tokens += header_tokens
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"CONTEXT {i+1}:\n{chunk.content}\n\n"
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            # Check if adding this chunk would exceed token limit
            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.info(f"Context truncated at {i} chunks due to token limit")
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        final_context = context_header + "".join(context_parts)
        
        logger.info(f"Optimized context: {total_tokens} tokens, {len(context_parts)} chunks")
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
    
    def retrieve_and_rank(
        self,
        query: str,
        use_query_expansion: bool = True,
        top_k: int = 5
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Main retrieval method using enhanced OpenAI-based approach
        Returns optimized context and retrieved chunks
        """

        # Use enhanced retrieval with OpenAI embeddings
        chunks = self.enhanced_retrieval(query, top_k * 2)

        # Apply relevance boosting
        for chunk in chunks:
            boost = self._calculate_relevance_boost(chunk, query)
            chunk.relevance_score = chunk.score + boost

        # Re-rank with boosted scores
        final_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)[:top_k]

        # Optimize context
        optimized_context = self.context_optimization(final_chunks, query)

        return optimized_context, final_chunks

# Example usage and testing
if __name__ == "__main__":
    # This would be used in the main application
    pass
