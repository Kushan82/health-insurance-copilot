"""
Conversation Manager - Tracks conversation history for context-aware responses
Enables follow-up questions and conversational RAG
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
import time


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    timestamp: float


class ConversationManager:
    """
    Manages conversation history and context
    
    Features:
    - Remembers previous queries and answers
    - Resolves follow-up questions using context
    - Limits context window to recent turns
    """
    
    def __init__(self, max_history: int = 5):
        """
        Args:
            max_history: Maximum number of turns to remember
        """
        self.history: List[ConversationTurn] = []
        self.max_history = max_history
        logger.info(f"Conversation manager initialized (max_history: {max_history})")
    
    def add_turn(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        confidence: float
    ):
        """Add a conversation turn to history"""
        turn = ConversationTurn(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self.history.append(turn)
        
        # Trim history if exceeds max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(f"Added turn to history (total: {len(self.history)})")
    
    def get_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context as formatted string
        
        Args:
            num_turns: Number of recent turns to include
        
        Returns:
            Formatted context string
        """
        if not self.history:
            return ""
        
        recent_turns = self.history[-num_turns:]
        
        context_lines = ["Previous conversation:"]
        for i, turn in enumerate(recent_turns, 1):
            context_lines.append(f"\nQ{i}: {turn.query}")
            context_lines.append(f"A{i}: {turn.answer[:200]}...")  # First 200 chars
        
        context_lines.append("\n---\n")
        
        return "\n".join(context_lines)
    
    def resolve_follow_up(self, query: str) -> str:
        """
        Resolve follow-up question by adding context from history
        
        Examples:
        - "What about maternity?" → "What about maternity in [previous policy]?"
        - "How much does it cost?" → "What is the premium for [previous policy]?"
        
        Args:
            query: Current user query
        
        Returns:
            Resolved query with context
        """
        if not self.history:
            return query
        
        query_lower = query.lower().strip()
        
        # Detect pronouns and demonstratives (indicators of follow-up)
        follow_up_indicators = [
            'it', 'this', 'that', 'they', 'them', 'its',
            'what about', 'how about', 'and', 'also'
        ]
        
        is_follow_up = any(indicator in query_lower.split()[:3] for indicator in follow_up_indicators)
        
        if not is_follow_up:
            return query
        
        # Get previous query context
        last_turn = self.history[-1]
        
        # Simple resolution: append previous query context
        resolved = f"{query} (context: {last_turn.query})"
        
        logger.info(f"Resolved follow-up: '{query}' → '{resolved}'")
        
        return resolved
    
    def clear(self):
        """Clear conversation history"""
        count = len(self.history)
        self.history.clear()
        logger.info(f"Conversation history cleared ({count} turns)")
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        return {
            'total_turns': len(self.history),
            'max_history': self.max_history,
            'avg_confidence': sum(t.confidence for t in self.history) / len(self.history) if self.history else 0
        }
