"""Extract specific facts from policy chunks"""
import re
from typing import Optional, Dict

class FactExtractor:
    """Extract structured facts like waiting periods, amounts"""
    
    def extract_waiting_period(self, text: str) -> Optional[Dict]:
        """Extract waiting period mentions"""
        patterns = [
            r'(?:waiting period|PED|pre-existing).*?(\d+)\s*(months?|years?)',
            r'(\d+)\s*(months?|years?).*?(?:waiting|continuous coverage)',
            r'(?:covered after|eligible after)\s*(\d+)\s*(months?|years?)',
            r'(?:thirty-six|thrity six)\s*months'  # Map to 36
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                unit = match.group(2).lower()
                
                # Normalize to months
                months = value * 12 if 'year' in unit else value
                
                return {
                    "value": value,
                    "unit": unit,
                    "months": months,
                    "text": match.group(0)
                }
        
        return None
    
    def extract_coverage_amount(self, text: str) -> Optional[Dict]:
        """Extract sum insured amounts"""
        patterns = [
            r'₹\s*(\d+(?:,\d+)*)\s*([LlCc]r?)?',
            r'Rs\.?\s*(\d+(?:,\d+)*)\s*([LlCc]r?)?',
        ]
        
        amounts = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                suffix = match.group(2) if len(match.groups()) > 1 else None
                if suffix:
                    suffix_lower = suffix.lower()
                    if 'l' in suffix_lower and 'cr' not in suffix_lower:
                        amount *= 100000  # Lakh
                    elif 'cr' in suffix_lower:
                        amount *= 10000000  # Crore
                
                amounts.append({
                    "amount": amount,
                    "formatted": match.group(0)
                })
        
        return amounts[0] if amounts else None