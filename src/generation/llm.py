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

EVAL_SYSTEM_PROMPT = """
You are an independent evaluator. Your task is to rate the truthfulness and alignment of the answer based ONLY on the provided context.
Compare the question, the context, and the proposed answer.
Rate the confidence of the answer on a scale from 0.0 to 1.0:
- 1.0: The answer is completely supported by the context and directly answers the question.
- 0.5: The answer is partially supported, or contains some uncertainty.
- 0.0: The answer is not supported by the context, or states that it cannot be found.
Provide ONLY a single float value between 0.0 and 1.0. Do not include any other text, explanation, or markdown.
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
            confidence = self._calculate_confidence(question, answer, context)
            return answer, confidence

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error during response generation: {str(e)}", 0.0



    def _calculate_confidence(self, question: str, answer: str, context: list) -> float:
        """Calculate confidence score using a combination of retrieval score, reranker score, and LLM self-evaluation"""
        try:
            # 1. Retrieval Score (average similarity score of retrieved nodes)
            retrieval_scores = []
            for node in context:
                score = getattr(node, "score", None)
                if score is not None:
                    retrieval_scores.append(score)
                elif hasattr(node, "metadata") and node.metadata and "retrieval_score" in node.metadata:
                    retrieval_scores.append(node.metadata["retrieval_score"])
                else:
                    retrieval_scores.append(1.0)
            avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.5

            # 2. Reranker/Lexical Overlap Score (keyword overlap between question and contexts)
            avg_reranker_score = self._calculate_reranker_score(question, context)

            # 3. LLM Self-Evaluation
            context_str = ""
            for i, chunk in enumerate(context):
                context_str += f"[Source {i+1}]: {chunk.text}\n"

            llm_eval_score = self._llm_self_evaluation(question, context_str, answer)

            # Weighted average: 0.3 * Retrieval + 0.3 * Reranker + 0.4 * LLM Eval
            confidence = (0.3 * avg_retrieval_score) + (0.3 * avg_reranker_score) + (0.4 * llm_eval_score)

            logger.info(
                f"Confidence components - Retrieval: {avg_retrieval_score:.4f}, "
                f"Reranker: {avg_reranker_score:.4f}, LLM Eval: {llm_eval_score:.4f} -> Total: {confidence:.4f}"
            )

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

    def _calculate_reranker_score(self, question: str, context: list) -> float:
        """Compute a lexical overlap score (proportion of question words found in context) as a reranker score"""
        try:
            stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
                "is", "was", "were", "are", "of", "by", "this", "that", "it", "he", "she",
                "they", "we", "i", "you", "my", "your", "his", "her", "their", "our", "us", "them"
            }
            # Clean and tokenize question into content words
            q_words = [w.strip(".,;:?!()\"'[]").lower() for w in question.split()]
            q_content_words = [
                w for w in q_words if w and w not in stop_words and len(w) > 2
            ]

            if not q_content_words:
                return 0.5

            # Find the best matching chunk (max overlap score)
            max_overlap_ratio = 0.0
            for chunk in context:
                chunk_text_lower = chunk.text.lower()
                matches = sum(1 for w in q_content_words if w in chunk_text_lower)
                overlap_ratio = matches / len(q_content_words)
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio

            return max_overlap_ratio
        except Exception as e:
            logger.warning(f"Error calculating reranker score: {e}")
            return 0.5

    def _llm_self_evaluation(self, question: str, context_str: str, answer: str) -> float:
        """Ask LLM to evaluate the confidence/groundedness of the generated answer against the context"""
        try:
            prompt = f"Question: {question}\nContext:\n{context_str}\nAnswer: {answer}\nConfidence Score:"
            eval_ans = ""

            # Try Ollama client
            try:
                import ollama
                response = ollama.chat(
                    model=self.config.OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": 0.0,  # Use 0 temp for deterministic evaluation
                        "num_predict": 10,   # Only need a few tokens for a float
                    },
                )
                eval_ans = response.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning(f"Ollama library self-eval failed: {e}. Trying HTTP...")

            # Fallback to HTTP
            if not eval_ans:
                url = f"{self.config.OLLAMA_BASE_URL}/api/chat"
                payload = {
                    "model": self.config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 10,
                    },
                    "stream": False,
                }
                res = requests.post(url, json=payload, timeout=15)
                if res.status_code == 200:
                    eval_ans = res.json().get("message", {}).get("content", "").strip()

            if eval_ans:
                # Clean up the output to extract float
                import re
                match = re.search(r"(\d+(\.\d+)?)", eval_ans)
                if match:
                    val = float(match.group(1))
                    return max(0.0, min(1.0, val))

            # If all LLM calls fail, fallback to a lexical overlap check between answer and context
            return self._calculate_fallback_word_overlap(answer, context_str)
        except Exception as e:
            logger.warning(f"Error in LLM self-evaluation: {e}")
            return 0.7

    def _calculate_fallback_word_overlap(self, answer: str, context_str: str) -> float:
        """Fallback word overlap percentage of content words between answer and context"""
        try:
            stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
                "is", "was", "were", "are", "of", "by", "this", "that", "it", "he", "she",
                "they", "we", "i", "you", "my", "your", "his", "her", "their", "our", "us", "them"
            }
            words = [w.strip(".,;:?!()\"'[]").lower() for w in answer.split()]
            content_words = [w for w in words if w and w not in stop_words and len(w) > 2]
            if not content_words:
                return 0.5
            context_lower = context_str.lower()
            matches = sum(1 for w in content_words if w in context_lower)
            return matches / len(content_words)
        except Exception:
            return 0.5
