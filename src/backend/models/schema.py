from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class SentenceSpan(BaseModel):
    """Represents a sentence with its character offsets in the original text."""
    text: str
    start: int
    end: int


class CaseItem(BaseModel):
    """Represents a case from the counseling Q&A corpus."""
    id: int
    context: str
    response: str
    response_sentences: List[SentenceSpan] = Field(default_factory=list)


class QuoteRef(BaseModel):
    """Reference to a specific quote from a case."""
    case_id: int
    sent_id: int = Field(default=0)
    
    model_config = {
        "validate_assignment": False  # Allow field modification after creation
    }


class Highlight(BaseModel):
    """A highlighted sentence from a case response."""
    sent_id: int
    text: str
    start: int
    end: int
    score: float


class SearchResult(BaseModel):
    """A case with highlights returned from search."""
    case_id: int
    context: str
    response: str  # Added full response
    score: float
    highlights: List[Highlight]


class SearchRequest(BaseModel):
    """Request for case search."""
    query: str
    k: int = 3
    include_highlights: bool = False


class Skill(BaseModel):
    """A CBT micro-skill with supporting quotes."""
    name: str
    quote_refs: List[QuoteRef]


class CopingSuggestion(BaseModel):
    """A coping suggestion with supporting quotes."""
    label: str
    quote_refs: List[QuoteRef]


class Goal(BaseModel):
    """A SMART goal with supporting quotes."""
    label: str
    quote_refs: List[QuoteRef]


class DeciderOutput(BaseModel):
    """Output from the LLM decider."""
    crisis_level: Literal["none", "mild", "moderate", "high"]
    skills: List[Skill]
    coping_suggestions: List[CopingSuggestion]
    goals: List[Goal]
    notes: str = Field(..., max_length=40)

    @field_validator("notes")
    def validate_notes_length(cls, v):
        words = v.split()
        if len(words) > 40:
            return " ".join(words[:40])
        return v


class CoachRequest(BaseModel):
    """Request for coaching."""
    query: str
    case_ids: List[int]


class CoachResponse(BaseModel):
    """Response from the coach service."""
    answer_markdown: Optional[str] = None  # Full markdown response from synthesis
    validation: Optional[str] = None
    reflection: Optional[str] = None
    coping_suggestions: List[str] = []
    goals: List[str] = []
    resources: List[str] = []
    citations: List[Dict[str, Any]] = []
    refused: bool = False
    refusal_reason: Optional[str] = None
    crisis_detected: bool = False 


class SynthesisOutput(BaseModel):
    """Output from the synthesis LLM call."""
    answer_markdown: str              # full final text, markdown allowed
    citations: List[QuoteRef]         # case_id, sent_id (and optional start,end) 