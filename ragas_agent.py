from dotenv import load_dotenv
load_dotenv()

import os

from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, SemanticSimilarity

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(groq_llm)

st_model = SentenceTransformer("all-MiniLM-L6-v2")


class SentenceTransformerEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return st_model.encode(texts).tolist()

    def embed_query(self, text):
        return st_model.encode(text).tolist()


ragas_embeddings = LangchainEmbeddingsWrapper(
    SentenceTransformerEmbeddings()
)

semantic_similarity_metric = SemanticSimilarity(embeddings=ragas_embeddings)

run_config = RunConfig(
    timeout=180,
    max_workers=1,
    max_retries=2,
)

def _local_similarity(text_a: str, text_b: str) -> float:
    dataset = Dataset.from_dict({"response": [text_a], "reference": [text_b]})
    result = evaluate(
        dataset,
        metrics=[semantic_similarity_metric],
        run_config=run_config,
    )
    score = float(result.to_pandas().iloc[0]["semantic_similarity"])
    return round(max(0.0, min(1.0, score)), 2)

def ragas_evaluate(
    question,
    answer,
    reference_answer,
    contexts,
):

    
    relevance = _local_similarity(answer, question)
    accuracy = _local_similarity(answer, reference_answer)
    completeness = _local_similarity(reference_answer, answer)

    
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "ground_truth": [reference_answer],
        "contexts": [contexts],
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    faithful = float(result.to_pandas().iloc[0]["faithfulness"])

    if faithful >= 0.85:
        hallucination = "Low"
    elif faithful >= 0.60:
        hallucination = "Medium"
    else:
        hallucination = "High"

    overall = (accuracy + relevance + completeness + faithful) / 4

    return {
        "accuracy": round(accuracy, 2),
        "relevance": round(relevance, 2),
        "completeness": round(completeness, 2),
        "faithfulness": round(faithful, 2),
        "hallucination_risk": hallucination,
        "overall_score": round(overall, 2)
    }