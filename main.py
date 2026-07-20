import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from retrieval_agent import retrieve_context
from relevance_agent import judge_relevance
from accuracy_agent import judge_accuracy
from hallucination_agent import detect_hallucination
from database import init_db, save_evaluation, get_history


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

init_db()


@app.get("/")
def home():
    return {
        "message": "AI Response Quality Evaluator API Running"
    }


def extract_pdf_text(file_bytes: bytes, max_chars: int = 3000) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + " "
        if len(text) >= max_chars:
            break
    return text.strip()[:max_chars]


# Numeric proxy for hallucination risk, used only to compute the overall score
_RISK_TO_SCORE = {"Low": 1.0, "Medium": 0.6, "High": 0.2}


@app.post("/evaluate")
async def evaluate(
    question: str = Form(...),
    answer: str = Form(...),
    reference_answer: Optional[str] = Form(None),
    source_pdf: Optional[UploadFile] = File(None),
):

    retrieved_context = retrieve_context(question, top_k=2)

    used_source_pdf = False
    if source_pdf is not None:
        pdf_bytes = await source_pdf.read()
        pdf_text = extract_pdf_text(pdf_bytes)
        if pdf_text:
            retrieved_context.append(pdf_text)
            used_source_pdf = True

    # --- Relevance Judge Agent ---
    relevance_result = judge_relevance(question, answer)

    # --- Accuracy Judge Agent ---
    accuracy_result = judge_accuracy(
        answer=answer,
        reference_answer=reference_answer,
        contexts=retrieved_context,
    )

    # --- Hallucination Detection Agent ---
    hallucination_result = detect_hallucination(
        answer=answer,
        contexts=retrieved_context,
    )

    hallucination_score_proxy = _RISK_TO_SCORE.get(hallucination_result["hallucination_risk"], 0.5)

    overall_score = round(
        (relevance_result["relevance_score"] + accuracy_result["accuracy_score"] + hallucination_score_proxy) / 3,
        2,
    )

    evaluation = {
        "relevance": relevance_result["relevance_score"],
        "relevance_reasoning": relevance_result["reasoning"],
        "accuracy": accuracy_result["accuracy_score"],
        "accuracy_evidence": accuracy_result["evidence"],
        "accuracy_supporting_excerpt": accuracy_result["supporting_excerpt"],
        "hallucination_risk": hallucination_result["hallucination_risk"],
        "hallucination_flagged_statements": hallucination_result["flagged_statements"],
        "hallucination_reasoning": hallucination_result["reasoning"],
        "overall_score": overall_score,
        "used_reference_answer": accuracy_result["used_reference_answer"],
    }

    save_evaluation(
        question=question,
        answer=answer,
        reference_answer=reference_answer,
        used_source_pdf=used_source_pdf,
        retrieved_context=retrieved_context,
        evaluation=evaluation,
    )

    return {
        "status": "success",
        "question": question,
        "retrieved_context": retrieved_context,
        "used_source_pdf": used_source_pdf,
        "evaluation": evaluation,
    }


@app.get("/history")
def history(limit: int = 20):
    return {
        "status": "success",
        "history": get_history(limit)
    }