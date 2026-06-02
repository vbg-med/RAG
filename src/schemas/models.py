from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    text: str
    source: str
    page: str
    type: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: float


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str
