import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 환경 변수 로드
load_dotenv()

# ==========================================
# 0. 데이터 정의 (이미지 기반 수집 API 5종)
# ==========================================
API_REGISTRY = {
    "agri_idea_prod": {
        "name": "농산업사업화 아이디어 제품정보",
        "description": "농산업 분야의 사업화 아이디어와 관련된 제품 정보를 검색합니다. 농업 창업 아이디어 제품화 사례를 찾을 때 사용합니다.",
        "url": "http://apis.data.go.kr/B552895/itg_nadbiz_id_prd_info_gw/getItgNadbizIdPrdInfoListGw",
        "params": ["pageNo", "numOfRows"]
    },
    "agri_startup_prod": {
        "name": "농식품 창업제품 정보",
        "description": "농식품 분야에서 실제 창업으로 이어진 제품들의 상세 정보를 조회합니다. 농식품 스타트업 제품 라인업 확인에 적합합니다.",
        "url": "http://apis.data.go.kr/B552895/itg_agfd_sttp_prd_info_gw/getItgAgfdSttpPrdInfoListGw",
        "params": ["pageNo", "numOfRows"]
    },
    "agri_excellent_tech": {
        "name": "우수기술사업화알림정보",
        "description": "한국농업기술진흥원에서 인증한 우수 농업 기술 및 사업화 성공 사례 알림 정보를 조회합니다. 기술 이전 및 우수 기술 분석에 사용합니다.",
        "url": "http://apis.data.go.kr/B552895/itg_gotech_biz_noti_info_gw/getItgGotechBizNotiInfoListGw",
        "params": ["pageNo", "numOfRows"]
    },
    "agri_patent_search": {
        "name": "신착특허정보검색",
        "description": "농업 및 식품 분야의 최신 등록/출원된 신착 특허 정보를 검색합니다. 최신 농업 기술 트렌드나 특허 동향을 파악할 때 유효합니다.",
        "url": "http://apis.data.go.kr/B552895/itg_new_ptnt_info_srch_gw/getItgNewPtntInfoSrchListGw",
        "params": ["pageNo", "numOfRows", "searchKeyword"]
    },
    "agri_biz_prod": {
        "name": "사업품종소개",
        "description": "정부나 진흥원에서 추진하는 농업 사업 대상 품종 및 작물 정보를 소개합니다. 어떤 작물이나 품종이 사업화 대상인지 확인할 때 씁니다.",
        "url": "http://apis.data.go.kr/B552895/itg_biz_knd_intro_gw/getItgBizKndIntroListGw",
        "params": ["pageNo", "numOfRows"]
    }
}

# 로컬 임베딩 모델 초기화
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ==========================================
# 1단계: API 하나 호출 검증 (단일 API 펑션)
# ==========================================
def call_public_api(api_key: str, api_id: str, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """공공데이터포털 REST API 호출 공통 함수"""
    if api_id not in API_REGISTRY:
        return {"error": f"Unknown API ID: {api_id}"}
    
    api_info = API_REGISTRY[api_id]
    
    # 기본 파라미터 세팅
    params = {
        "serviceKey": api_key,
        "pageNo": 1,
        "numOfRows": 3,
        "_type": "json"  # JSON 응답 강제 (지원 안 할 시 XML 파싱 필요)
    }
    if extra_params:
        params.update(extra_params)
        
    try:
        response = requests.get(api_info["url"], params=params, timeout=10)
        # 공공데이터포털은 에러 발생 시에도 200 OK에 XML/JSON 에러 메시지를 보낼 때가 많음
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Response is not JSON", "raw": response.text[:200]}
        else:
            return {"error": f"HTTP Status {response.status_code}", "raw": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# ==========================================
# LLM 통신 및 토큰 추적 유틸리티
# ==========================================
def query_llm(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    """OpenAI 호환 API를 호출하고 응답과 대략적인 입력 토큰 수를 반환"""
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL_NAME")
    
    # 간단한 글자 수 기반 토큰 추정 (또는 API 응답의 usage 사용 가능)
    # 여기서는 프롬프트 입력 토큰 절약 비교를 위해 글자 수 기반 러프 추정치를 기록하거나 
    # API 실제 usage 값을 리턴받도록 유도합니다.
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0 # 일관된 tool 선택을 위해 0으로 고정
    }
    
    try:
        res = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=15)
        res_json = res.json()
        content = res_json['choices'][0]['message']['content']
        input_tokens = res_json.get('usage', {}).get('prompt_tokens', len(system_prompt) + len(user_prompt) // 3)
        return content, input_tokens
    except Exception as e:
        return f"LLM Error: {str(e)}", 0

# ==========================================
# 2단계: 모든 도구를 컨텍스트에 포함 (방식 A)
# ==========================================
def run_approach_a(user_question: str) -> Tuple[str, int]:
    """모든 API 명세를 프롬프트에 때려 넣는 전통적 방식"""
    tools_string = json.dumps(API_REGISTRY, ensure_ascii=False, indent=2)
    
    system_prompt = f"""당신은 사용자의 질문을 해결하기 위해 어떤 공공데이터 API를 호출해야 하는지 결정하는 에이전트입니다.
아래 제공된 JSON 형태의 API 목록 중에서 질문과 가장 일치하는 API 키(ID)를 선택하고, 파라미터가 필요하다면 추출하세요.

[API 목록]
{tools_string}

[출력 포맷 가이드]
반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 생략합니다.
{{
    "api_id": "선택한_API_키",
    "reason": "선택한 이유 짧게"
}}
"""
    return query_llm(system_prompt, user_question)

# ==========================================
# 3단계: RAG 기반 Dynamic Toolset 선택 (방식 B)
# ==========================================
# 3단계 사전 작업: 레지스트리 임베딩 빌드
api_keys_list = list(API_REGISTRY.keys())
api_descriptions = [f"{v['name']} : {v['description']}" for v in API_REGISTRY.values()]
api_embeddings = embedding_model.encode(api_descriptions, convert_to_tensor=False)

def run_approach_b(user_question: str, top_k: int = 2, threshold: float = 0.3) -> Tuple[str, int]:
    """RAG로 관련 API만 필터링하여 프롬프트에 올리는 방식"""
    # 1. 질문 임베딩 및 유사도 계산
    query_emb = embedding_model.encode(user_question, convert_to_tensor=False)
    
    scores = []
    for emb in api_embeddings:
        # 코사인 유사도 계산
        dot_prod = np.dot(query_emb, emb)
        norm_q = np.linalg.norm(query_emb)
        norm_e = np.linalg.norm(emb)
        score = dot_prod / (norm_q * norm_e) if norm_q * norm_e > 0 else 0
        scores.append(score)
        
    # 2. Top-K 및 임계값 필터링
    sorted_idx = np.argsort(scores)[::-1]
    filtered_registry = {}
    
    for idx in sorted_idx[:top_k]:
        if scores[idx] >= threshold:
            key = api_keys_list[idx]
            filtered_registry[key] = API_REGISTRY[key]
            
    # 만약 검색된 도구가 하나도 없다면 가볍게 전체에서 top_1 강제 지정하거나 예외 처리
    if not filtered_registry:
        key = api_keys_list[sorted_idx[0]]
        filtered_registry[key] = API_REGISTRY[key]

    tools_string = json.dumps(filtered_registry, ensure_ascii=False, indent=2)
    
    system_prompt = f"""당신은 사용자의 질문을 해결하기 위해 어떤 공공데이터 API를 호출해야 하는지 결정하는 에이전트입니다.
아래 제공된 검색된 API 목록 중에서 질문과 가장 일치하는 API 키(ID)를 선택하세요.

[검색된 제한적 API 목록]
{tools_string}

[출력 포맷 가이드]
반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 생략합니다.
{{
    "api_id": "선택한_API_키",
    "reason": "선택한 이유 짧게"
}}
"""
    return query_llm(system_prompt, user_question)

# ==========================================
# 4단계: 비교 평가 자동화 수행 (eval)
# ==========================================
if __name__ == "__main__":
    service_key = os.getenv("DATA_GO_KR_SERVICE_KEY")
    
    print("=== 1단계: 단일 API 실제 호출 검증 ===")
    test_run = call_public_api(service_key, "agri_patent_search", {"searchKeyword": "드론"})
    print(f"호출 결과 요약: {str(test_run)[:200]}...\n")
    
    # 평가용 데이터셋 구축 (유사 단어 혼동 유도)
    eval_dataset = [
        {"question": "최근에 새로 나온 농업 관련 특허 출원 기술이 뭐가 있는지 찾아줘.", "expected": "agri_patent_search"},
        {"question": "청년들이 농업으로 창업해서 만든 실제 가공식품이나 상품 종류 리스트가 필요해.", "expected": "agri_startup_prod"},
        {"question": "진흥원에서 선정한 사업화 성공 우수 기술 목록이랑 공고 정보를 보고싶어.", "expected": "agri_excellent_tech"},
        {"question": "새로운 농산업 아이디어를 제품화시킨 사례나 기획 정보가 있나요?", "expected": "agri_idea_prod"},
        {"question": "정부 지원 사업 대상인 쌀이나 배 같은 농산물 품종 소개 데이터가 필요해.", "expected": "agri_biz_prod"},
    ]
    
    report_card = []
    
    print("=== 4단계: 방식 A vs 방식 B 비교 테스트 수행 ===")
    for idx, item in enumerate(eval_dataset):
        q = item["question"]
        expected = item["expected"]
        
        # 방식 A 수행
        res_a, tokens_a = run_approach_a(q)
        try:
            pred_a = json.loads(res_a).get("api_id")
        except:
            pred_a = "PARSE_ERROR"
            
        # 방식 B 수행
        res_b, tokens_b = run_approach_b(q, top_k=2, threshold=0.4)
        try:
            pred_b = json.loads(res_b).get("api_id")
        except:
            pred_b = "PARSE_ERROR"
            
        report_card.append({
            "idx": idx + 1,
            "question": q[:15] + "...",
            "expected": expected,
            "pred_a": pred_a,
            "tokens_a": tokens_a,
            "acc_a": 1 if pred_a == expected else 0,
            "pred_b": pred_b,
            "tokens_b": tokens_b,
            "acc_b": 1 if pred_b == expected else 0,
        })
        print(f"테스트 {idx+1} 완료.")

    # 결과 표 출력 출력 포맷 가공
    print("\n=== 최종 비교 결과 표 ===")
    print(f"{'번호':<4}|{'질문 요약':<12}|{'정답 API':<20}|{'방식A 결과':<20}|{'A토큰':<6}|{'방식B 결과':<20}|{'B토큰':<6}")
    print("-" * 95)
    total_tokens_a = 0
    total_tokens_b = 0
    total_acc_a = 0
    total_acc_b = 0
    
    for r in report_card:
        total_tokens_a += r["tokens_a"]
        total_tokens_b += r["tokens_b"]
        total_acc_a += r["acc_a"]
        total_acc_b += r["acc_b"]
        print(f"{r['idx']:<4}|{r['question']:<12}|{r['expected']:<20}|{r['pred_a']:<20}|{r['tokens_a']:<6}|{r['pred_b']:<20}|{r['tokens_b']:<6}")
        
    print("-" * 95)
    print(f"평균/합계 정확도 -> 방식 A: {total_acc_a/len(eval_dataset)*100:.1f}% | 방식 B: {total_acc_b/len(eval_dataset)*100:.1f}%")
    print(f"총 입력 토큰 소모량 -> 방식 A: {total_tokens_a} 토큰 | 방식 B: {total_tokens_b} 토큰")
    print(f"토큰 절감률: {(total_tokens_a - total_tokens_b)/total_tokens_a*100:.1f}% 절감 완료.")