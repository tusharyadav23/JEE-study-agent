"""JEE study agent: retrieval (FAISS) + answer generation (OpenAI-compatible LLM)."""
import re
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app import config
from app.ingest import VECTORSTORE_DIR, create_vectorstore

NO_ANSWER_EN = (
    "I couldn't find reliable information on that in my JEE study material. "
    "Try rephrasing, or ask about JEE syllabus, formulas, exam pattern or preparation strategy."
)
NO_ANSWER_HI = (
    "मुझे अपनी JEE अध्ययन सामग्री में इसका भरोसेमंद उत्तर नहीं मिला। "
    "कृपया प्रश्न दोबारा पूछें, या JEE सिलेबस, फ़ॉर्मूले, परीक्षा पैटर्न या तैयारी की रणनीति के बारे में पूछें।"
)

SYSTEM_PROMPT = """You are a friendly, accurate JEE (Main & Advanced) study assistant.
Answer ONLY using the context provided. If the context does not contain the answer, say you don't know.
Reply in the same language as the student's question (Hindi or English; Hinglish -> reply in Hindi).
Keep answers concise and clear. Keep formulas and technical terms in standard notation."""


def is_hindi(text: str) -> bool:
    """True if the text contains Devanagari characters."""
    return bool(re.search(r"[\u0900-\u097F]", text))


class JEEAgent:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )

        if not (VECTORSTORE_DIR / "index.faiss").exists():
            print("Vector store not found - building it now...")
            create_vectorstore(self.embeddings)

        self.vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            self.embeddings,
            # Safe here: we only ever load an index that we created ourselves.
            allow_dangerous_deserialization=True,
        )

        self.llm = None
        if config.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
            )

    # ------------------------------------------------------------------ retrieval
    def retrieve_with_scores(
        self, question: str, k: Optional[int] = None, threshold: Optional[float] = None
    ):
        """Top-k chunks, with weak matches (distance > threshold) filtered out."""
        k = k or config.TOP_K
        threshold = config.SCORE_THRESHOLD if threshold is None else threshold

        results = self.vectorstore.similarity_search_with_score(question, k=k)
        return [(doc, float(score)) for doc, score in results if float(score) <= threshold]

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(question, k)]

    # ------------------------------------------------------------------ answering
    def answer(self, question: str) -> dict:
        hindi = is_hindi(question)
        hits = self.retrieve_with_scores(question)

        if not hits:
            return {
                "answer": NO_ANSWER_HI if hindi else NO_ANSWER_EN,
                "sources": [],
                "grounded": False,
            }

        context = "\n\n---\n\n".join(doc.page_content for doc, _ in hits)
        sources = [
            {
                "source": doc.metadata.get("source", "unknown").split("/")[-1],
                "score": round(score, 3),
                "snippet": doc.page_content[:200].replace("\n", " "),
            }
            for doc, score in hits
        ]

        if self.llm is None:
            # No API key configured: return the best retrieved passage (extractive mode).
            return {
                "answer": hits[0][0].page_content.strip(),
                "sources": sources,
                "grounded": True,
                "mode": "extractive (set OPENAI_API_KEY for generated answers)",
            }

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        reply = self.llm.invoke(messages)
        return {"answer": reply.content, "sources": sources, "grounded": True, "mode": "llm"}
