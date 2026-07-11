from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval_agent import retrieve_context
from ragas_agent import ragas_evaluate


app = FastAPI(
    title="AI Response Quality Evaluator",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    reference_answer: str



@app.get("/")
def home():
    return {
        "message": "AI Response Quality Evaluator API Running"
    }


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):



    retrieved_context = retrieve_context(
        request.question,
        top_k=2
    )


    evaluation = ragas_evaluate(
        question=request.question,
        answer=request.answer,
        reference_answer=request.reference_answer,
        contexts=retrieved_context
    )


    return {

        "status": "success",

        "question": request.question,

        "retrieved_context": retrieved_context,

        "evaluation": evaluation

    }