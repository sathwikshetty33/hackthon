from typing import List, Dict, Any
from pydantic import BaseModel, HttpUrl



class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class DetailedAnswerResponse(BaseModel):
    answers: List[Dict[str, Any]]
