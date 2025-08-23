import os
import logging
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# .env 파일에서 API 키 로드
load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- 상태 정의 ---
class AgentState(TypedDict):
    original_query: str
    query_type: Literal["KCD", "CANCER_REG", "UNKNOWN"] # 쿼리 유형 추가
    expanded_terms: List[str]

# --- LLM 및 출력 파서 정의 ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, thinking_budget=0)

# 1. 의도 분류를 위한 파서
class QueryClassifier(BaseModel):
    query_type: Literal["KCD", "CANCER_REG", "UNKNOWN"] = Field(
        description="""사용자 쿼리의 의도를 'KCD', 'CANCER_REG', 'UNKNOWN' 세 가지 유형으로 분류합니다.

- 'KCD': **한국표준질병사인분류(KCD) 코드나 질병명에 대한 질문입니다.** 주로 특정 질병, 증상, 약어에 해당하는 KCD 코드를 찾거나, KCD 코드의 의미를 묻는 경우입니다.(영어의 경우 대소문자를 구분하지 않습니다.)
  - 예시: "폐렴 KCD 코드", "C50.9 코드", "상세불명의 위염", "DM", "AKI", "sepsis", "VRE infection", "mm", "MM", "aki", "dm", "pjp", "pcp", "rabbit syndrome", "hcc" 등

- 'CANCER_REG': **암 등록을 위한 특정 정보(TNM 병기, SEER 요약 병기, 분화도, 침범 범위, 편측성 등)에 대한 코드나 기준을 묻는 질문입니다.** 주로 병리 결과지 내용이나 암 관련 특정 용어에 대한 코드를 찾는 경우입니다.
  - 예시: "위암 perigastric fat tissue invasion seer코드", "유방암 분화도 3등급", "편측성을 입력해야하는 암 부위", "TNM staging for colon cancer", "Pathology report: Colon, proximal ascending, endoscopic submucosal dissection: Tubular adenoma with focal high grade dysplasia"

- 'UNKNOWN': **KCD 코드나 암 등록과 직접적인 관련이 없는 일반적인 의학 질문, 또는 의도가 불분명한 쿼리입니다.**
  - 예시: "코로나 증상", "당뇨병 치료 방법", "혈압약 종류", "건강검진 항목 추천"
"""
    )

# 2. KCD 쿼리 확장을 위한 파서
class KcdQueryExpansion(BaseModel):
    expanded_terms: List[str] = Field(
        description="KCD 코드 검색에 최적화된 검색어 목록 (원본 쿼리 포함)"
    )

class CancerRegQueryExpansion(BaseModel):
    expanded_terms: List[str] = Field(
        description="암등록 코드 검색에 최적화된 검색어 목록 (원본 쿼리 포함)"
    )

class GeneralQueryExpansion(BaseModel):
    expanded_terms: List[str] = Field(
        description="일반적인 검색어 목록"
    )

# --- 노드 함수 정의 ---

def classify_query_node(state: AgentState):
    """사용자 쿼리의 의도를 'KCD', 'CANCER_REG', 'UNKNOWN'으로 분류합니다."""
    logger.info(f"Executing node: classify_query for query: '{state['original_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자 쿼리의 의도를 분석하는 전문가입니다. 쿼리가 KCD 코드에 대한 질문인지, 암등록 관련 질문인지, 아니면 둘 다 아닌지 판단해주세요."),
        ("human", "다음 쿼리의 의도를 분류해주세요: {query}")
    ])
    
    chain = prompt | llm.with_structured_output(QueryClassifier)
    result = chain.invoke({"query": state["original_query"]})
    
    state['query_type'] = result.query_type
    logger.info(f"Query classified as: {result.query_type}")
    return state

def expand_kcd_query_node(state: AgentState):
    """KCD 관련 쿼리를 확장합니다."""
    logger.info(f"Executing node: expand_kcd_query for query: '{state['original_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
역할 : 당신은 한국표준질병사인분류(KCD) 및 의학용어 전문가이자 검색 쿼리 향상 전문가입니다.

**1단계: 쿼리 분석**
- 입력이 의료 약어(영문 또는 국문)일 경우, KCD 진단 코드의 맥락에서 가장 일반적으로 사용되는 공식적인 풀네임(Full term)을 반환합니다.
약어가 여러 의미를 가질 수 있을 때는 가장 흔히 사용되는 진단명을 기준으로 합니다. (예: 'DM' -> 'Diabetes Mellitus', 'HEP' -> 'Hepatic Encephalopathy', 'HCC' -> 'Hepatocellular Carcinoma', 'hCCA' -> 'Hilar Cholangiocarcinoma', 'AKI' -> 'Acute Kidney Injury', 'MM' -> 'Multiple Myeloma', 'pcp' -> 'Pneumocystis Pneumonia', 'pjp' -> 'Pneumocystis jirovecii pneumonia', )
- 약어가 아닌, Full-Term또는 문장이라면 원본 쿼리를 그대로 사용합니다. (예: '상세불명의 위염' -> '상세불명의 위염')
- 영어의 경우 대소문자를 구분하지 않습니다.

**2단계: 검색어 확장**
- **1단계 결과**를 기반으로 검색 효율을 높일 검색어를 생성합니다.
- 원본쿼리가 의학약어 또는 의학용어라면 기반 용어(Full-Term 또는 원본 쿼리)가 한국어이면 영어로, 영어이면 한국어로 된 동의어/유사어/관련어를 1개 추가합니다.
- **최종 결과에는 반드시 원본 쿼리가 포함되어야 합니다.**

**확장된 쿼리는 최대 3개입니다**

예시:
- 입력: "MM" -> 출력: ["MM", "Multiple Myeloma", "다발성 골수종"]
- 입력: "mm" -> 출력: ["mm", "Multiple Myeloma", "다발성 골수종"]
- 입력: "폐렴 코드 알려줘" -> 출력: ["폐렴 코드 알려줘", "pneumonia", "폐렴"]
- 입력: "VRE infection" -> 출력: ["VRE infection", "Vancomycin-resistant Enterococcus infection", "반코마이신 내성 장알균 감염"]
- 입력: "MRONJ" -> 출력: ["MRONJ", "Medication-Related Osteonecrosis of the Jaw", "약물 관련 턱뼈 괴사"]
- 입력: "mronj" -> 출력: ["mronj", "Medication-Related Osteonecrosis of the Jaw", "약물 관련 턱뼈 괴사"]
"""),
        ("human", "다음 쿼리를 KCD 검색에 맞게 확장해주세요: {query}")
    ])
    
    chain = prompt | llm.with_structured_output(KcdQueryExpansion)
    result = chain.invoke({"query": state["original_query"]})
    
    if state['original_query'] not in result.expanded_terms:
        result.expanded_terms.insert(0, state['original_query'])
        
    state['expanded_terms'] = result.expanded_terms
    logger.info(f"Expanded KCD query terms: {result.expanded_terms}")
    return state

def expand_cancer_reg_query_node(state: AgentState):
    """암등록 관련 쿼리를 확장합니다."""
    logger.info(f"Executing node: expand_cancer_reg_query for query: '{state['original_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
역할 : 암등록(Cancer Registry) 관련 쿼리 확장 전문가입니다.

주어진 쿼리가 암등록 정보를 요청하는것으로 판단된다면
-쿼리에 약어가 있는 경우가 아니면 원본 쿼리만 반환합니다

쿼리에 약어가 있다면 약어를 Full-Term으로 변환하여 쿼리확장을 1개만 추가합니다다

예시:
- 입력: "PTC" -> 출력: ["PTC", "Papillary Thyroid Carcinoma"]
- 입력: "soft tissue cancer의 skin invasion의 seer코드 알려줘" -> 출력: ["soft tissue cancer의 skin invasion의 seer코드 알려줘]
"""),
        ("human", "다음 쿼리를 암등록 검색에 맞게 확장해주세요: {query}")
    ])
    
    chain = prompt | llm.with_structured_output(CancerRegQueryExpansion)
    result = chain.invoke({"query": state['original_query']})
    
    if state['original_query'] not in result.expanded_terms:
        result.expanded_terms.insert(0, state['original_query'])
        
    state['expanded_terms'] = result.expanded_terms
    logger.info(f"Expanded Cancer Registry query terms: {result.expanded_terms}")
    return state

def general_expansion_node(state: AgentState):
    """일반 쿼리에 포함된 의학 약어를 확장합니다."""
    logger.info(f"Executing node: general_expansion for query: '{state['original_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 의학용어 전문가입니다.
주어진 쿼리에서 의학 약어를 찾아 Full-Term으로 확장해주세요.
- 약어가 없다면 원본 쿼리만 포함된 리스트를 반환합니다.
- 약어가 있다면, 원본 쿼리와 Full-Term을 포함한 리스트를 반환합니다.
- 최종 결과에는 항상 원본 쿼리가 포함되어야 합니다.
- 문장 형태의 쿼리라면 문장의 의미는 유지하면서 핵심 키워드를 추출또는 요약하여 1개 추가합니다. (문장에 "코드", "환자", "KCD" 등의 단어가 포함되어 있다면 해당 단어는 제외하고 핵심 내용만 추출합니다. 예: '코로나 감염 후 폐렴 KCD 코드' -> '코로나19 감염 후 폐렴')

예시:
- 입력: "hCCA 환자 치료" -> 출력: ["hCCA 환자 치료", "Hilar Cholangiocarcinoma 환자 치료"]
- 입력: "코로나 증상" -> 출력: ["코로나 증상"]
- 입력: "외부에서 일하면서 땀을 많이 흘림. 그 뒤로 온 몸이 아프고 쥐가 나고 두통, 어지럼증 있음" -> 출력: ["외부에서 일하면서 땀을 많이 흘림. 그 뒤로 온 몸이 아프고 쥐가 나고 두통, 어지럼증 있음", "과도한 땀 흘림 후 전신 통증, 근육 경련, 두통, 어지럼증"]
"""),
        ("human", "다음 쿼리를 분석하고 필요시 확장해주세요: {query}")
    ])
    
    chain = prompt | llm.with_structured_output(GeneralQueryExpansion)
    result = chain.invoke({"query": state["original_query"]})
    
    if state['original_query'] not in result.expanded_terms:
        result.expanded_terms.insert(0, state['original_query'])
        
    state['expanded_terms'] = result.expanded_terms
    logger.info(f"Expanded general query terms: {result.expanded_terms}")
    return state

# --- 그래프 구성 ---
workflow = StateGraph(AgentState)

# 1. 노드 추가
workflow.add_node("classify_query", classify_query_node)
workflow.add_node("expand_kcd_query", expand_kcd_query_node)
workflow.add_node("expand_cancer_reg_query", expand_cancer_reg_query_node)
workflow.add_node("general_expansion", general_expansion_node)

# 2. 진입점 설정
workflow.set_entry_point("classify_query")

# 3. 조건부 엣지(분기) 설정
def decide_next_node(state: AgentState):
    logger.info(f"Deciding next node based on query type: {state['query_type']}")
    if state['query_type'] == "KCD":
        return "expand_kcd_query"
    elif state['query_type'] == "CANCER_REG":
        return "expand_cancer_reg_query"
    else:
        return "general_expansion"

workflow.add_conditional_edges(
    "classify_query",
    decide_next_node,
    {
        "expand_kcd_query": "expand_kcd_query",
        "expand_cancer_reg_query": "expand_cancer_reg_query",
        "general_expansion": "general_expansion"
    }
)

# 4. 종료점 설정
workflow.add_edge("expand_kcd_query", END)
workflow.add_edge("expand_cancer_reg_query", END)
workflow.add_edge("general_expansion", END)

# 5. 그래프 컴파일
app = workflow.compile()

# # --- 테스트 실행 ---
# if __name__ == "__main__":
#     # 로깅을 테스트하기 위한 간단한 설정
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     queries = [
#         "MM",                                       # KCD 예상
#         "폐렴 KCD 코드 알려줘",                       # KCD 예상
#         "soft tissue cancer skin invasion seer code", # 암등록 예상
#         "위암 분화도",                                # 암등록 예상
#         "코로나 증상"                                 # 일반 예상
#     ]

#     for query in queries:
#         print(f"\n{'='*30}\n테스트 쿼리: '{query}'\n{'='*30}")
#         inputs = {"original_query": query}
#         final_state = app.invoke(inputs)
        
#         print("\n--- 최종 상태 결과 ---")
#         for key, value in final_state.items():
#             print(f"  {key}: {value}")
#         print("-" * 25)