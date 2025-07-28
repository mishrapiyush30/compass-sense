import logging
import json
import time
import random
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import requests
from pydantic import ValidationError

from ..models.schema import CaseItem, DeciderOutput, CoachResponse, QuoteRef, SynthesisOutput
from ..services.safety_service import SafetyService

logger = logging.getLogger(__name__)


class CoachService:
    """Service for coaching with synthesis LLM approach."""
    
    def __init__(
        self, 
        cases: List[CaseItem],
        safety_service: SafetyService,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the coach service.
        
        Args:
            cases: List of all cases
            safety_service: Safety service for crisis detection
            config: Configuration parameters
        """
        self.cases = cases
        self.safety_service = safety_service
        
        # Default configuration
        self.config = {
            "llm_api_url": "https://api.openai.com/v1/chat/completions",
            "llm_api_key": None,  # Must be provided
            "llm_model": "gpt-4o-mini",  # Faster model with good JSON adherence
            "llm_temperature": 0,
            "llm_max_tokens": 450,  # Slightly larger to allow multi-paragraph reply
            "llm_timeout": 20.0,  # Seconds (increased from 6.0)
            "resources": [
                "Crisis Text Line: Text HOME to 741741 (US)",
                "National Suicide Prevention Lifeline: 988 or 1-800-273-8255",
                "SAMHSA's National Helpline: 1-800-662-HELP (4357)",
                "7 Cups: Free online chat with trained listeners at 7cups.com"
            ]
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        logger.info("Initialized coach service")
    
    async def _call_llm_synthesis(self, query: str, cases: List[Dict[str, Any]]) -> Optional[SynthesisOutput]:
        """
        Ask the LLM to write a full, supportive response using ONLY the provided case responses,
        and to include citations for each key claim.
        
        Args:
            query: User query
            cases: List of cases with full responses
            
        Returns:
            SynthesisOutput object or None if failed
        """
        if not self.config["llm_api_key"]:
            logger.error("LLM API key not provided")
            return None
        
        logger.info("==================== LLM SYNTHESIS CALL START ====================")
        logger.info(f"Calling LLM with query: {query}")
        logger.info(f"Using {len(cases)} cases as context")
        
        # Build compact context: include full responses for the selected cases,
        # and instruct to cite the EXACT sentences being referenced.
        cases_text = []
        for case in cases:
            # Include full responses and context
            case_text = f"Case {case['case_id']}\nContext: {case['context']}\nResponse:\n{case.get('response','')}"
            cases_text.append(case_text.strip())
        cases_str = "\n\n".join(cases_text)

        system_prompt = (
            "You are a careful, supportive coach. You must use ONLY the content from the provided case responses.\n"
            "Write a clear, human-friendly reply (150-250 words) with a supportive tone, organized in short paragraphs.\n"
            "Rules:\n"
            "1) Do not invent facts. Use phrasing consistent with the provided responses.\n"
            "2) Include citations for your key claims and suggestions. Cite by case_id and sent_id.\n"
            "   Example citation forms inside the answer: [C101:S3], [C87:S1]\n"
            "3) Keep suggestions actionable and specific (e.g., a brief breathing exercise, a short walk, a small journaling step).\n"
            "4) No clinical/medical directives. Add a brief safety note at the end.\n"
            "5) For each citation, identify the specific sentence number in the response. Sentences are 0-indexed.\n"
            "Output STRICT JSON with fields:\n"
            "{\n"
            '  "answer_markdown": "<your multi-paragraph markdown with inline [Ccase:Sent] citations>",\n'
            '  "citations": [{"case_id":101,"sent_id":3}, {"case_id":87,"sent_id":1}]\n'
            "}\n"
        )

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Cases (use only this material, and cite precise sentences by sent_id):\n{cases_str}\n\n"
            "Write the final answer with inline [Ccase:Sent] citations, then return the JSON."
        )
        
        logger.info("System prompt:")
        logger.info(system_prompt)
        logger.info("User prompt:")
        logger.info(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)

        headers = {
            "Authorization": f"Bearer {self.config['llm_api_key']}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": self.config["llm_model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config["llm_temperature"],
            "max_tokens": self.config["llm_max_tokens"],
            "response_format": {"type": "json_object"},
        }

        try:
            logger.info(f"Sending request to {self.config['llm_api_url']}")
            resp = requests.post(
                self.config["llm_api_url"], 
                headers=headers, 
                json=body, 
                timeout=self.config["llm_timeout"],
                verify=False  # Disable SSL verification for testing
            )
            
            if resp.status_code != 200:
                logger.error(f"LLM error {resp.status_code}: {resp.text[:300]}")
                logger.info("==================== LLM SYNTHESIS CALL FAILED ====================")
                return None
            
            payload = resp.json()
            text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"LLM raw response: {text}")
            
            try:
                data = json.loads(text)
                logger.info(f"Parsed JSON: {json.dumps(data, indent=2)}")
                
                # Create SynthesisOutput object
                synthesis_output = SynthesisOutput(**data)
                logger.info("Successfully created SynthesisOutput")
                logger.info("==================== LLM SYNTHESIS CALL SUCCESS ====================")
                return synthesis_output
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                logger.info("==================== LLM SYNTHESIS CALL FAILED (PARSE ERROR) ====================")
                return None
        except Exception as e:
            logger.error(f"LLM synthesis request failed: {e}")
            logger.info("==================== LLM SYNTHESIS CALL FAILED (REQUEST ERROR) ====================")
            return None
    
    def _light_verify(self, synthesis: SynthesisOutput) -> Tuple[bool, str]:
        """
        Minimal safety/grounding checks:
        - At least 2 citations.
        - Each citation points to an existing sentence.
        - Optional: Light n-gram overlap check between cited sentences and the answer.
        """
        logger.info("Running light verification on synthesis output")
        logger.info(f"Raw answer with citations: {synthesis.answer_markdown}")
        logger.info(f"Citations: {synthesis.citations}")
        
        if len(synthesis.citations) < 2:
            logger.warning("Verification failed: Need at least 2 citations")
            return False, "Need at least 2 citations."

        # Fix any out-of-range sentence indices and check if cases exist
        fixed_citations = []
        for ref in synthesis.citations:
            try:
                case = next((c for c in self.cases if c.id == ref.case_id), None)
                if not case:
                    logger.warning(f"Verification failed: Bad citation: unknown case {ref.case_id}")
                    return False, f"Bad citation: unknown case {ref.case_id}"
                
                # If sentence index is out of range, clamp it to the valid range
                if ref.sent_id < 0:
                    logger.warning(f"Fixing negative sentence index: case {ref.case_id} sent {ref.sent_id} -> 0")
                    ref.sent_id = 0
                elif ref.sent_id >= len(case.response_sentences):
                    valid_idx = len(case.response_sentences) - 1
                    logger.warning(f"Fixing out-of-range sentence index: case {ref.case_id} sent {ref.sent_id} -> {valid_idx}")
                    ref.sent_id = valid_idx
                
                fixed_citations.append(ref)
            except Exception as e:
                logger.error(f"Citation lookup error: {e}")
                return False, "Citation lookup error"
        
        # Replace the original citations with fixed ones
        synthesis.citations = fixed_citations
        
        # Optional: Light n-gram overlap check
        try:
            # Simple tokenization function
            def tokenize(text):
                # Convert to lowercase, remove punctuation, split on whitespace
                import re
                return re.sub(r'[^\w\s]', '', text.lower()).split()
            
            # Extract n-grams from text
            def get_ngrams(tokens, n):
                return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            
            # Check if there's reasonable overlap between answer and cited sentences
            answer_tokens = tokenize(synthesis.answer_markdown)
            
            # For each citation, check if there's some n-gram overlap with the answer
            for ref in synthesis.citations:
                case = next((c for c in self.cases if c.id == ref.case_id), None)
                if case and 0 <= ref.sent_id < len(case.response_sentences):
                    cited_text = case.response_sentences[ref.sent_id].text
                    cited_tokens = tokenize(cited_text)
                    
                    # Get bigrams from both texts
                    answer_bigrams = set(get_ngrams(answer_tokens, 2))
                    cited_bigrams = set(get_ngrams(cited_tokens, 2))
                    
                    # Calculate overlap
                    if len(cited_bigrams) > 0:
                        overlap = len(answer_bigrams.intersection(cited_bigrams)) / len(cited_bigrams)
                        logger.info(f"Citation overlap for case {ref.case_id}, sent {ref.sent_id}: {overlap:.2f}")
                        
                        # If overlap is too low, log a warning but don't fail verification
                        # This is a soft check that could be made stricter in the future
                        if overlap < 0.1:  # Very minimal threshold
                            logger.warning(f"Low citation overlap ({overlap:.2f}) for case {ref.case_id}, sent {ref.sent_id}")
        except Exception as e:
            # Don't fail verification if the overlap check itself fails
            logger.error(f"Error in overlap check: {e}")

        logger.info("Verification passed")
        return True, ""
    
    async def coach(self, query: str, cases: List[Dict[str, Any]]) -> CoachResponse:
        """
        Generate a coaching response using synthesis approach.
        
        Args:
            query: User query
            cases: List of cases with highlights
            
        Returns:
            CoachResponse object
        """
        start_time = time.time()
        logger.info(f"Generating coaching response for query: '{query}'")
        
        # 1) Crisis front gate
        if self.safety_service.detect_crisis(query):
            logger.warning("Crisis language detected in query")
            return CoachResponse(
                refused=True,
                refusal_reason="Crisis language detected. This system is not equipped to handle crisis situations. Please seek appropriate support.",
                resources=[],
                crisis_detected=True
            )
        
        # 2) LLM synthesis (single call)
        logger.info("Calling LLM synthesis")
        synthesis = await self._call_llm_synthesis(query, cases)
        if not synthesis:
            logger.warning("LLM synthesis failed, returning fallback response")
            return CoachResponse(
                refused=True,
                refusal_reason="Temporarily unable to generate a response. Please try again.",
                resources=[],
                crisis_detected=False
            )
        
        # 3) Light verification
        logger.info("Running light verification")
        ok, reason = self._light_verify(synthesis)
        if not ok:
            logger.warning(f"Light verification failed: {reason}")
            return CoachResponse(
                refused=True,
                refusal_reason=f"Insufficient grounding: {reason}",
                resources=[],
                crisis_detected=False
            )
        
        # 4) Build citations payload (start/end from the referenced sentences)
        logger.info("Building citations payload")
        citations = []
        for ref in synthesis.citations:
            try:
                case = next((c for c in self.cases if c.id == ref.case_id), None)
                if not case:
                    logger.warning(f"Case not found: {ref.case_id}")
                    continue
                    
                if ref.sent_id < 0 or ref.sent_id >= len(case.response_sentences):
                    logger.warning(f"Sentence index out of range: case_id={ref.case_id}, sent_id={ref.sent_id}")
                    continue
                    
                sent = case.response_sentences[ref.sent_id]
                citations.append({
                    "case_id": ref.case_id,
                    "sent_id": ref.sent_id,
                    "start": sent.start,
                    "end": sent.end,
                    "text": sent.text
                })
            except Exception as e:
                logger.error(f"Error building citation: {e}")
        
        # 5) Return the final answer (as provided by the LLM)
        logger.info(f"Generated coaching response in {time.time() - start_time:.2f}s")
        return CoachResponse(
            answer_markdown=synthesis.answer_markdown,
            citations=citations,
            resources=[],
            refused=False,
            crisis_detected=False
        ) 