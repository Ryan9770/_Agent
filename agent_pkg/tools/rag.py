"""
RAG 도구: search_docs (임베딩 기반 로컬 문서 검색).

임베딩 모델은 공용 embeddings 모듈에서 가져온다(중복 로딩 방지).
문서 적재·임베딩 사전계산은 import 시 한 번 일어난다.
"""

import os
import glob

import numpy as np

from .. import config
from ..embeddings import encode
from .registry import tool


def load_and_chunk(docs_dir=config.DOCS_DIR,
                   chunk_size=config.CHUNK_SIZE,
                   overlap=config.CHUNK_OVERLAP):
    """폴더 안 텍스트 파일을 일정 글자 수 조각으로 자른다(overlap만큼 겹침)."""
    chunks = []
    paths = glob.glob(os.path.join(docs_dir, "*.txt")) + \
            glob.glob(os.path.join(docs_dir, "*.md"))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            chunks.append({"source": os.path.basename(path), "text": chunk})
            start += chunk_size - overlap
    return chunks


# ── 시작 시 한 번: 문서 적재 + 임베딩 사전 계산 ──
DOC_CHUNKS = load_and_chunk()
print(f"[RAG] {len(DOC_CHUNKS)}개 조각 적재 완료 (from {config.DOCS_DIR})")

_texts = [c["text"] for c in DOC_CHUNKS]
DOC_EMBEDDINGS = encode(_texts) if _texts else np.zeros((0, 384))
print(f"[RAG] 임베딩 사전 계산 완료: shape={DOC_EMBEDDINGS.shape}")


@tool(
    name="search_docs",
    description="Search the user's local document collection for relevant "
                "passages. Use this when the user asks about information that "
                "may be in their documents.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search keywords"},
            "top_k": {"type": "integer", "description": "how many passages, default 3"},
        },
        "required": ["query"],
    },
)
def search_docs(query: str, top_k: int = 3) -> str:
    """질문을 임베딩해 의미적으로 가장 가까운 조각을 코사인 유사도로 찾는다.
    단, 최고 유사도가 임계값보다 낮으면 '관련 문서 없음'을 담백하게 알린다."""
    if not DOC_CHUNKS:
        return "No documents loaded. Put .txt/.md files in ./docs"

    q_vec = encode([query])[0]
    scores = DOC_EMBEDDINGS @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]

    best_score = float(scores[top_idx[0]]) if len(top_idx) else -1.0
    if best_score < config.RAG_MIN_SCORE:
        return (f"No strongly matching documents for '{query}' "
                f"(best similarity {best_score:.3f}).")

    out = []
    for i in top_idx:
        ch = DOC_CHUNKS[i]
        out.append(f"[출처: {ch['source']} | 유사도: {scores[i]:.3f}]\n{ch['text']}")
    return "\n\n---\n\n".join(out)