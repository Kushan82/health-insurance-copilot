"""
Query Rewriter - Transforms user queries for better retrieval
Handles abbreviations, typos, and informal language
"""
from typing import Dict, List
from loguru import logger


class QueryRewriter:
    """Rewrites and clarifies user queries"""
    
    def __init__(self):
        # Common insurance abbreviations
        self.abbreviations = {
            'ped': 'pre-existing disease',
            'si': 'sum insured',
            'ncb': 'no claim bonus',
            'opd': 'outpatient department',
            'ipd': 'inpatient department',
            'icu': 'intensive care unit',
            'ayush': 'ayurveda yoga unani siddha homeopathy',
            'co-pay': 'copayment',
            'sub-limit': 'sub limit',
            'covid': 'covid-19 coronavirus',
        }
        
        # Common typos and variations
        self.corrections = {
            'maturnity': 'maternity',
            'pregnency': 'pregnancy',
            'hosptial': 'hospital',
            'insurence': 'insurance',
            'exlusion': 'exclusion',
            'premimum': 'premium',
        }
        
        # Policy name variations
        self.policy_mappings = {
            'optima': 'optima restore',
            'medicare': 'medicare plus',
            'care supreme': 'care supreme',
            'care ultimate': 'care ultimate',
            'bajaj': 'my health care plan',
        }
        
    def rewrite(self, query: str) -> Dict[str, str]:
        """
        Rewrite query for better retrieval
        
        Returns:
            {
                'original': original query,
                'rewritten': improved query,
                'expansions': list of alternate phrasings
            }
        """
        query_lower = query.lower().strip()
        rewritten = query_lower
        
        # Step 1: Expand abbreviations
        for abbr, full in self.abbreviations.items():
            if abbr in rewritten.split():
                rewritten = rewritten.replace(abbr, f"{abbr} {full}")
                logger.debug(f"Expanded abbreviation: {abbr} → {full}")
        
        # Step 2: Fix typos
        for typo, correct in self.corrections.items():
            if typo in rewritten:
                rewritten = rewritten.replace(typo, correct)
                logger.debug(f"Corrected typo: {typo} → {correct}")
        
        # Step 3: Normalize policy names
        for variant, canonical in self.policy_mappings.items():
            if variant in rewritten and canonical not in rewritten:
                rewritten = rewritten.replace(variant, canonical)
        
        # Step 4: Generate alternate phrasings
        expansions = self._generate_expansions(rewritten)
        
        result = {
            'original': query,
            'rewritten': rewritten,
            'expansions': expansions
        }
        
        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return result
    
    def _generate_expansions(self, query: str) -> List[str]:
        """Generate alternate phrasings of the query"""
        expansions = []
        
        # Pattern 1: "What is X?" → "Explain X" / "Tell me about X"
        if query.startswith('what is'):
            topic = query.replace('what is', '').strip('?').strip()
            expansions.append(f"explain {topic}")
            expansions.append(f"tell me about {topic}")
            expansions.append(f"details of {topic}")
        
        # Pattern 2: "Does policy cover X?" → "Is X covered?" / "X coverage"
        if 'does' in query and 'cover' in query:
            # Extract topic between "cover" and end
            parts = query.split('cover')
            if len(parts) > 1:
                topic = parts[1].strip('?').strip()
                expansions.append(f"is {topic} covered")
                expansions.append(f"{topic} coverage")
                expansions.append(f"{topic} benefits")
        
        # Pattern 3: "waiting period" → "waiting time" / "eligibility period"
        if 'waiting period' in query:
            expansions.append(query.replace('waiting period', 'waiting time'))
            expansions.append(query.replace('waiting period', 'eligibility period'))
        
        # Pattern 4: Add question variations
        if not query.endswith('?'):
            expansions.append(f"what is {query}")
        
        return expansions[:3]  # Return top 3 expansions


class MultiQueryGenerator:
    """Generates multiple query variations for better recall"""
    
    def __init__(self):
        self.templates = {
            'waiting_period': [
                "What is the waiting period for {topic}?",
                "How long is the waiting time for {topic}?",
                "When can I claim {topic}?",
            ],
            'coverage': [
                "Does the policy cover {topic}?",
                "Is {topic} included in coverage?",
                "What are the benefits for {topic}?",
            ],
            'limit': [
                "What is the limit for {topic}?",
                "What are the sub-limits for {topic}?",
                "Is there a cap on {topic}?",
            ],
            'comparison': [
                "Compare {topic} between policies",
                "Which policy has better {topic}?",
                "Difference in {topic} coverage",
            ],
        }
    
    def generate(self, query: str, query_type: str = 'general') -> List[str]:
        """
        Generate multiple query variations
        
        Args:
            query: Original query
            query_type: Type of query (waiting_period, coverage, limit, comparison)
        
        Returns:
            List of query variations (including original)
        """
        variations = [query]  # Always include original
        
        # Detect query type if not specified
        if query_type == 'general':
            query_type = self._detect_query_type(query)
        
        # Extract topic from query
        topic = self._extract_topic(query, query_type)
        
        # Generate variations using templates
        if query_type in self.templates and topic:
            for template in self.templates[query_type]:
                variations.append(template.format(topic=topic))
        
        # Add policy-specific variations if policy detected
        policy = self._detect_policy(query)
        if policy:
            variations.append(f"{query} in {policy}")
            variations.append(f"{policy} {topic}")
        
        logger.info(f"Generated {len(variations)} query variations")
        return variations[:5]  # Return top 5
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type from content"""
        query_lower = query.lower()
        
        if 'waiting' in query_lower or 'how long' in query_lower:
            return 'waiting_period'
        elif 'cover' in query_lower or 'include' in query_lower:
            return 'coverage'
        elif 'limit' in query_lower or 'cap' in query_lower or 'sub-limit' in query_lower:
            return 'limit'
        elif 'compare' in query_lower or 'vs' in query_lower or 'better' in query_lower:
            return 'comparison'
        
        return 'general'
    
    def _extract_topic(self, query: str, query_type: str) -> str:
        """Extract main topic from query"""
        query_lower = query.lower()
        
        # Remove common question words
        stop_words = ['what', 'is', 'the', 'are', 'does', 'policy', 'cover', 'in', 'for', 'a', 'an']
        words = [w for w in query_lower.split() if w not in stop_words]
        
        # Look for key insurance terms
        key_terms = ['maternity', 'ayush', 'room rent', 'ped', 'pre-existing', 
                     'restore', 'day care', 'ambulance', 'copay', 'deductible']
        
        for term in key_terms:
            if term in query_lower:
                return term
        
        # Return remaining words as topic
        return ' '.join(words[:3]) if words else ''
    
    def _detect_policy(self, query: str) -> str:
        """Detect policy name from query"""
        query_lower = query.lower()
        
        policies = [
            'optima restore', 'optima secure',
            'care supreme', 'care ultimate',
            'medicare plus', 'medicare premier',
            'my health care plan'
        ]
        
        for policy in policies:
            if policy in query_lower:
                return policy
        
        return ''


class QueryExpander:
    """Expands queries with domain-specific synonyms"""
    
    def __init__(self):
        # Domain-specific synonym mappings
        self.synonyms = {
            'waiting period': ['waiting time', 'cooling period', 'eligibility period', 'qualification period'],
            'pre-existing disease': ['ped', 'pre-existing condition', 'prior illness', 'chronic condition'],
            'room rent': ['room charges', 'room category', 'accommodation charges', 'hospital room'],
            'maternity': ['pregnancy', 'childbirth', 'delivery', 'maternal care'],
            'day care': ['daycare', 'day-care procedure', 'short procedure', 'outpatient procedure'],
            'cover': ['coverage', 'included', 'benefits', 'covered'],
            'exclusion': ['not covered', 'excluded', 'limitation', 'restriction'],
            'sum insured': ['si', 'coverage amount', 'insured amount', 'policy limit'],
            'premium': ['cost', 'price', 'payment', 'contribution'],
            'claim': ['reimbursement', 'settlement', 'payout', 'compensation'],
        }
    
    def expand(self, query: str) -> Dict[str, List[str]]:
        """
        Expand query with synonyms
        
        Returns:
            {
                'original': original query,
                'expanded_terms': dict of term -> synonyms found in query,
                'expanded_query': query with all synonym variations
            }
        """
        query_lower = query.lower()
        expanded_terms = {}
        
        # Find matching terms and their synonyms
        for term, syns in self.synonyms.items():
            if term in query_lower:
                expanded_terms[term] = syns
        
        logger.info(f"Expanded {len(expanded_terms)} terms with synonyms")
        
        return {
            'original': query,
            'expanded_terms': expanded_terms,
            'all_variants': self._generate_variants(query, expanded_terms)
        }
    
    def _generate_variants(self, query: str, expanded_terms: Dict[str, List[str]]) -> List[str]:
        """Generate query variants with synonym substitution"""
        variants = [query]
        
        for term, synonyms in expanded_terms.items():
            # Create one variant per synonym
            for syn in synonyms[:2]:  # Limit to 2 synonyms per term
                variant = query.lower().replace(term, syn)
                if variant not in variants:
                    variants.append(variant)
        
        return variants[:5]  # Return top 5 variants
