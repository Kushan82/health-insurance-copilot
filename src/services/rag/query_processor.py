from typing import List, Dict, Optional
from loguru import logger
import re

class QueryProcessor:
    """Preprocess and expand user queries for better retrieval"""
    
    def __init__(self):
        # Insurance domain synonyms
        self.synonyms = {
            'waiting period': ['waiting time', 'eligibility period', 'qualifying period'],
            'pre-existing': ['pre existing', 'preexisting', 'PED'],
            'maternity': ['pregnancy', 'childbirth', 'maternal'],
            'room rent': ['room charges', 'room category', 'accommodation'],
            'coverage': ['benefits', 'cover', 'covered', 'what is covered'],
            'exclusion': ['not covered', 'limitations', 'restrictions']
        }
        
        # Policy name normalization
        self.policy_names = {
            'optima restore': ['HDFC Optima Restore', 'Optima Restore'],
            'care supreme': ['Care Supreme', 'Care Health Supreme'],
            'care ultimate': ['Care Ultimate', 'Ultimate Care'],
            'medicare plus': ['Medicare Plus', 'Max Bupa Medicare Plus'],
            'medicare premier': ['Medicare Premier', 'Max Bupa Medicare Premier'],
            'optima secure': ['HDFC Optima Secure', 'Optima Secure']
        }
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process query and return enhanced version with metadata
        
        Returns:
            {
                'original': str,
                'processed': str,
                'expanded_terms': List[str],
                'detected_policy': str | None,
                'query_type': str  # 'policy_specific', 'general', 'comparison'
            }
        """
        query_lower = query.lower().strip()
        
        # Detect policy name
        detected_policy = self._detect_policy(query_lower)
        
        # Determine query type
        query_type = self._classify_query(query_lower)
        
        # Expand query with synonyms
        expanded_terms = self._expand_with_synonyms(query_lower)
        
        # Build processed query
        processed_query = self._build_processed_query(
            query_lower, 
            expanded_terms, 
            detected_policy
        )
        
        result = {
            'original': query,
            'processed': processed_query,
            'expanded_terms': expanded_terms,
            'detected_policy': detected_policy,
            'query_type': query_type
        }
        
        logger.info(f"Query processed: {result['query_type']} | Policy: {detected_policy}")
        return result
    
    def _detect_policy(self, query: str) -> Optional[str]:
        """Detect which policy is being asked about"""
        for policy_key, variants in self.policy_names.items():
            for variant in variants:
                if variant.lower() in query:
                    return policy_key
        return None
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for targeted retrieval"""
        if 'compare' in query or ' vs ' in query or ' versus ' in query:
            return 'comparison'
        elif self._detect_policy(query):
            return 'policy_specific'
        else:
            return 'general'
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with domain synonyms"""
        expanded = []
        for term, syns in self.synonyms.items():
            if term in query:
                expanded.extend(syns)
        return expanded
    
    def _build_processed_query(
        self, 
        query: str, 
        expanded_terms: List[str], 
        policy: Optional[str]
    ) -> str:
        """Build enhanced query string"""
        # Start with original
        parts = [query]
        
        # Add policy name variants
        if policy and policy in self.policy_names:
            parts.extend([p.lower() for p in self.policy_names[policy]])
        
        # Add top 2 expanded terms
        if expanded_terms:
            parts.extend(expanded_terms[:2])
        
        return ' '.join(parts)
