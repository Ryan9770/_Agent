"""
공용 임베딩 — 임베딩 모델을 한 곳에서 로드하고, 텍스트→벡터 변환을 제공한다.

rag.py(문서 검색)와 agent.py(의미 기반 중복 차단)가 모두 이 모듈을 쓴다.
모델은 import 시 한 번만 GPU에 로드된다(기존 동작 유지).
config만 의존하는 하위 계층이라 누구든 안전하게 import할 수 있다.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

print(f"[EMBED] 임베딩 모델 로딩 중: {config.EMBED_MODEL_NAME} ...")
_model = SentenceTransformer(config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE)
print(f"[EMBED] 임베딩 모델 로드 완료 (device: {_model.device})")


def encode(texts, normalize=True):
    """텍스트(또는 리스트)를 정규화된 벡터로 변환한다."""
    return _model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)


def cosine(vec_a, vec_b):
    """정규화된 두 벡터의 코사인 유사도(= 내적)."""
    return float(np.dot(vec_a, vec_b))