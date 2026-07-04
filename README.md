# AI Response Quality Evaluator Engine
 
A multi-agent LLM evaluation system that scores AI-generated responses across relevance, accuracy, hallucination, and completeness using a RAG pipeline grounded in TruthfulQA and SQuAD.
 
## Tech Stack
 
- **Frontend:** HTML/CSS/JS
- **Backend:** FastAPI
- **RAG:** LangChain
- **Embeddings:** Sentence-Transformers
- **Vector Store:** ChromaDB
- **Grounding Datasets:** TruthfulQA, SQuAD
- **Evaluation:** RAGAS, TruLens
  
## How It Works
 
1. User submits a question, AI response, and reference answer
2. Input is cleaned and validated
3. Relevant context is retrieved from ChromaDB
4. Four agents evaluate the response in parallel: relevance, accuracy, hallucination, completeness
5. Scores are aggregated into an overall result
6. A final report is generated with reasoning

