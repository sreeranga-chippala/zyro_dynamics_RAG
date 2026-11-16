# Zyro Dynamics Knowledge Assistant

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your PDFs to `data/` or upload them in the sidebar.
4. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Included RAG approaches

Chunking, Hugging Face embeddings, FAISS, dense similarity retrieval, MMR, BM25, hybrid retrieval, Reciprocal Rank Fusion, multi-query retrieval, HyDE, cross-encoder reranking, source/page citations, and abstention for unsupported questions.

## Important notes

- The app does not silently use outside knowledge to answer.
- Tune the relevance threshold against a labelled in-scope/out-of-scope evaluation set before treating it as a production gate.
- `Compare all strategies` retrieves with each baseline and combines candidate pools for inspection; the selected strategy controls the final answer path.
- The PDF documents are not bundled in this package. Add the original knowledge-base PDFs to `data/` or upload them at runtime.

## Corpus checks observed during preparation

The available file set contains `00_Company_Profile.pdf` and `02`–`09` policy documents naming Zyro Dynamics, but `01_Employee_Handbook.pdf` names Acrux Dynamics. The dashboard displays a warning for this conflict rather than silently reconciling it. A separate Travel & Expense PDF was not present in the available set, so add it if travel-policy questions must be answered.
