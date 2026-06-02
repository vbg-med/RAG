import logging
import requests
from src.config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a document QA assistant. IMPORTANT RULES:

1. ONLY answer using the provided context
2. If answer is not in context, say: "Not found in documents"
3. Always cite page numbers
4. For image interpretations, mention: "Based on chart/figure..."
5. Do NOT use outside knowledge
6. If uncertain, ask for clarification

Context will be provided below.
Answer concisely and factually.
"""


class LLMGenerator:
    """Interfaces with Ollama LLM to answer user queries using retrieved document contexts"""

    def __init__(self, config: Config):
        self.config = config

    def generate(self, question: str, context: list) -> tuple[str, float]:
        """Generate answer using Ollama and calculate confidence score"""
        if not context:
            return "Not found in documents", 0.0

        try:
            # Build context string
            context_str = ""
            for i, chunk in enumerate(context):
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", "N/A")
                type_ = chunk.metadata.get("type", "text")
                context_str += f"\n--- Source [{i + 1}]: {source} (Page {page}, Type {type_}) ---\n"
                context_str += chunk.text + "\n"

            prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
            answer = ""

            # 1. Attempt using Python client library
            try:
                import ollama

                response = ollama.chat(
                    model=self.config.OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": self.config.LLM_TEMPERATURE,
                        "num_predict": self.config.LLM_MAX_TOKENS,
                    },
                )
                answer = response.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning(
                    f"Ollama library chat failed: {e}. Trying direct HTTP endpoint..."
                )

            # 2. Fallback to direct HTTP endpoint (extremely robust)
            if not answer:
                url = f"{self.config.OLLAMA_BASE_URL}/api/chat"
                payload = {
                    "model": self.config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {
                        "temperature": self.config.LLM_TEMPERATURE,
                        "num_predict": self.config.LLM_MAX_TOKENS,
                    },
                    "stream": False,
                }
                res = requests.post(url, json=payload, timeout=60)
                if res.status_code == 200:
                    answer = res.json().get("message", {}).get("content", "").strip()
                else:
                    logger.error(
                        f"Ollama HTTP API returned status {res.status_code}: {res.text}"
                    )
                    raise Exception(f"Ollama API query failed: {res.text}")

            # Check for negative answers
            if not answer or "not found in documents" in answer.lower():
                return "Not found in documents", 0.0

            # Calculate grounded confidence
            confidence = self._calculate_confidence(answer, context_str)
            return answer, confidence

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error during response generation: {str(e)}", 0.0

    def _calculate_confidence(self, answer: str, context: str) -> float:
        """Calculate confidence score as overlap percentage of content words between answer and context"""
        try:
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "is",
                "was",
                "were",
                "are",
                "of",
                "by",
                "this",
                "that",
                "it",
                "he",
                "she",
                "they",
                "we",
                "i",
                "you",
                "my",
                "your",
                "his",
                "her",
                "their",
                "our",
                "us",
                "them",
            }

            # Clean and split answer into words
            words = [w.strip(".,;:?!()\"'[]").lower() for w in answer.split()]
            # Keep words that are longer than 2 characters and are not stop words
            content_words = [
                w for w in words if w and w not in stop_words and len(w) > 2
            ]

            if not content_words:
                return 0.5

            context_lower = context.lower()
            matches = sum(1 for w in content_words if w in context_lower)

            # Word-level overlap percentage
            confidence = matches / len(content_words)

            # Penalty for indicators of uncertainty
            lower_ans = answer.lower()
            if (
                "not sure" in lower_ans
                or "uncertain" in lower_ans
                or "unknown" in lower_ans
            ):
                confidence *= 0.5

            return max(0.1, min(1.0, confidence))
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return 0.75
