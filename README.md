A multi-agent LLM evaluation system that scores AI-generated responses across relevance, accuracy, hallucination, and completeness using a RAG pipeline grounded in TruthfulQA and SQuAD.

Tech Stack


Frontend: HTML/CSS/JS
Backend: FastAPI
RAG: LangChain
Embeddings: Sentence-Transformers
Vector Store: FAISS
Grounding Datasets: TruthfulQA, SQuAD
Evaluation: RAGAS, TruLens
Deployment: Render + Vercel


How It Works


User submits a question, AI response, and reference answer
Input is cleaned and validated
Relevant context is retrieved from FAISS
Four agents evaluate the response in parallel: relevance, accuracy, hallucination, completeness
Scores are aggregated into an overall result
A final report is generated with reasoning
