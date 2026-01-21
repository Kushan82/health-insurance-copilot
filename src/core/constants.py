"""Application constants"""

# Insurance Companies
INSURERS = [
    "bajaj_allianz",
    "hdfc_ergo",
    "tata_aig",
    "care_health"
]

# Policy Names
POLICIES = [
    "My Health Care Plan 1",
    "My Health Care Plan 6",
    "Optima Restore",
    "Optima Secure",
    "Medicare Plus",
    "Medicare Premier",
    "Care Supreme",
    "Care Ultimate"
]

# Cache Prefixes
CACHE_PREFIX_QUERY = "query:"
CACHE_PREFIX_SEMANTIC = "semantic:"
CACHE_PREFIX_RESPONSE = "response:"
CACHE_PREFIX_RETRIEVAL = "retrieval:"

# Monitoring Metrics
METRIC_LATENCY = "latency"
METRIC_CACHE_HIT = "cache_hit"
METRIC_CACHE_MISS = "cache_miss"
METRIC_TOKEN_USAGE = "token_usage"
METRIC_GUARDRAIL_VIOLATION = "guardrail_violation"

# System Prompts
SYSTEM_PROMPT = """You are an expert health insurance advisor helping users choose the best health insurance policy in India.

Your role:
- Ask clarifying questions to understand user needs (age, family structure, budget, health conditions)
- Compare policies objectively based on features, coverage, waiting periods, and value
- Explain insurance concepts in simple, easy-to-understand language
- Provide personalized recommendations with clear reasoning
- Be transparent about policy limitations and exclusions

Guidelines:
- Always be helpful, accurate, and empathetic
- Use examples to clarify complex concepts
- Never make guarantees about claim approval
- Recommend consulting policy documents for final decisions
- If unsure about something, acknowledge it honestly

Your goal is to help users make informed decisions about their health insurance."""

# File Extensions
ALLOWED_DOCUMENT_EXTENSIONS = [".pdf", ".docx", ".txt"]
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# API Response Messages
MSG_SUCCESS = "Operation completed successfully"
MSG_ERROR = "An error occurred"
MSG_NOT_FOUND = "Resource not found"
MSG_UNAUTHORIZED = "Unauthorized access"
MSG_INVALID_INPUT = "Invalid input provided"
