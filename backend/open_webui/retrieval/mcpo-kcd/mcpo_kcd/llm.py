from __future__ import annotations

import os
from typing import List, Optional
import logging


class LLMClient:
    """Thin wrapper for LLM calls with graceful fallbacks.

    - If OPENAI_API_KEY is present and openai package is installed, uses it.
    - Otherwise, returns heuristic results so the server works offline.
    """

    def __init__(self, model: str = "gpt-5-mini", base_url: Optional[str] = None, require_llm: Optional[bool] = None):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.require_llm = bool(int(os.getenv("MCPO_REQUIRE_LLM", "0"))) if require_llm is None else require_llm
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        try:
            self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
        except Exception:
            self.max_retries = 1
        self._client = None
        try:
            if self.api_key:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout)
                self.logger.info("LLM client initialized (model=%s, base_url=%s)", self.model, (self.base_url or "default"))
        except Exception as e:
            self._client = None
            self.logger.warning("Failed to initialize LLM client: %s", e)

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        if not self._client:
            if self.require_llm:
                raise RuntimeError("LLM required but not available. Set OPENAI_API_KEY.")
            # No LLM available and LLM not strictly required: return original only
            self.logger.debug("expand_query fallback (no LLM). Returning original only.")
            return [query]

        # Force strict JSON output for reliable parsing
        prompt = (
            "너는 의학 질의 확장 도우미야. 입력 질의에 포함된 약어를 풀어 쓰고, 임상의가 이해하기 쉬운 한국어 표현으로 재구성해.\n"
            "반드시 아래 형식의 JSON만 출력해. 그 외 설명/텍스트 금지.\n"
            "{\n  \"expansions\": [\"문장1\", \"문장2\", \"문장3\"]\n}\n"
            "제약:\n"
            "- 약어가 있을 때 최대3개 제시, 없으면 원문 그대로 1개만.\n"
            "- 중복/동의어 반복 금지.\n"
            f"원본 질의: {query}"
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="minimal",
                response_format={"type": "json_object"}
            )
            text = resp.choices[0].message.content
            import json
            data = json.loads(text)
            arr = data.get("expansions", []) if isinstance(data, dict) else []
            if not isinstance(arr, list):
                raise ValueError("Invalid JSON: 'expansions' must be a list")
            out: List[str] = []
            for s in arr:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
                if len(out) >= (max_expansions or 3):
                    break
            self.logger.debug("expand_query produced %d expansions", len(out))
            return out or [query]
        except Exception as e:
            self.logger.warning("expand_query failed; returning original. error=%s", e)
            return [query]

    def expand_and_classify(self, query: str, max_expansions: int = 3):
        """Single LLM call to expand and classify.

        Returns dict with keys: expanded_queries (List[str]), classifications (Dict[str, List[str]]),
        and optionally external_matches (List[str]) when any item category includes "external".
        """
        if not self._client:
            if self.require_llm:
                raise RuntimeError("LLM required but not available. Set OPENAI_API_KEY.")
            # graceful: just return original with empty categories
            return {"expanded_queries": [query], "classifications": {query: []}}

        instruction = (
"""
# 역할
- 본 어시스턴트는 한국표준질병사인분류(KCD)와 의학용어 관련 전문지식을 활용하여, 의학 용어 입력값을 분석하고, 전문 검색어 및 관련 용어를 확장해 제공합니다.
- 각 확장된 쿼리에 대해 카테고리(pathogen, resistance, external)를 분류하여 제공합니다

## 1단계: 입력 용어 분석 및 변환
- 입력된 용어가 의학 약어(영문 또는 국문)일 경우, KCD 진단 코드의 맥락에서 가장 일반적으로 사용되는 공식 용어(Full Term)로 변환합니다.
- 약어가 여러 의미를 가질 수 있을 때는 가장 흔히 사용되는 진단명을 기준으로 변환합니다.
  - 예시: 'DM' → 'Diabetes Mellitus', 'HEP' → 'Hepatic Encephalopathy', 'HCC' → 'Hepatocellular Carcinoma', 'hCCA' → 'Hilar Cholangiocarcinoma', 'AKI' → 'Acute Kidney Injury', 'MM' → 'Multiple Myeloma', 'pcp' → 'Pneumocystis Pneumonia', 'pjp' → 'Pneumocystis jirovecii pneumonia'
- 만약 약어가 아니거나 이미 Full-Term 또는 문장일 경우에는 원본 입력어를 그대로 사용합니다.
  - 예시: '상세불명의 위험' → '상세불명의 위험'
- 영문 입력어의 경우, 대소문자를 구분하지 않습니다.

## 2단계: 검색어 확장
- 1단계 결과를 기반으로 검색 효율을 높일 수 있는 확장 검색어를 생성합니다.
- 원본 용어가 의학 약어 또는 의학 용어인 경우, 기본 용어(Full-Term 또는 원본 표기)가 한국어면 영어로, 영어면 한국어로 변환된 동의어/유사어/관련어 중 하나를 추가하세요.
- 중요!! 최종 결과에는 반드시 원본 입력어가 포함되어야 합니다.

## 3단계: 쿼리 분류
- 각 확장 질의에 대해 필요한 카테고리(pathogen, resistance, external)를 분류합니다.
pathogen: 질병의 원인균에 대한 내용 포함 (예: 폐렴 원인균, 혈액배양 균 동정, 인플루엔자 바이러스, 결핵균, 노로바이러스, 대장균, 코로나바이러스 등)
resistance: 약제내성 등 내성과 관련된 내용 포함 (예: MRSA, VRE, ESBL, 카바페넴 내성(CRE), 약제 감수성, MIC 상승 등)
external: 어떤 질병, 사고 등의 원인, 외인에 대한 내용 포함 (예: 낙상/추락, 교통사고, 화상, 익사, 자해/자살 시도, 폭행, 외상, 합병증의 원인, 증상의 외인 등)
- categories는 위 세 값만 사용하고 필요 없으면 빈 배열.

## categories 값이 external을 포함할때 처리 규칙 추가
- external_matches 항목을 추가하여 반환합니다
- external_matches 값은 쿼리를 분석하여 아래의 내용 중 가장 가깝다고 판단되는 항목을 최대 3개 포함하여 반환합니다
운수사고에서 다친 보행자 (V01-V09)
운수사고에서 다친 자전거 탑승자(V10-V19)
운수사고에서 다친 모터사이클 탑승자(V20-V29)
운수사고에서 다친 삼륜자동차 탑승자(V30-V39
운수사고에서 다친 승용차 탑승자(V40-V49)
운수사고에서 다친 픽업트럭 또는 밴 탑승자 (V50-V59)
운수사고에서 다친 대형화물차 탑승자(V60-V69)
운수사고에서 다친 버스 탑승자(V70-V79)
기타 육상 운수사고 (V80-V89)
수상운수 사고 (V90-V94)
항공 및 우주 운수 사고 (V95-V97)
기타 및 상세 불명의 운수사고(V98-V99)
낙상 (W00-W19)
무생물성 기계적 힘에 노출 (W20-W49)
생물성 기계적 힘에 노출 (W50-W64)
우발적 익사 및 익수 (W65-W74)
호흡과 관련된 기타 불의의 위협(W75-W84)
전류, 방사선 및 극단적 기온 및 기압에의 노출(W85-W99)
연기, 불 및 불꽃에 노출(X00-X09)
열 및 가열된 물질과의 접촉(X10-X19)
독액성 동물 및 식물과의 접촉(X20-X29)
자연의 힘에 노출(X30-X39)
유독성 물질에 의한 불의의 중독 및 노출(X40-X49)
과잉노력, 여행 및 결핍(X50-X57)
기타 및 상세불명의 요인에 대한 사고피폭(X58-X59)
고의적 자해 (X60-X84)
가해 (X85-Y09)
의도 미확인 사건(Y10-Y34)
법적 개입 및 전쟁행위(Y35-Y36)
치료용으로 사용시 유해작용을 나타내는약물, 약제 및 생물학 물질(Y40-Y59)
외과적 및 내과적 치료 중 환자의 재난(Y60-Y69)
진단 및 치료용으로 사용시 유해사건과 관련된 의료장치(Y70-Y82)
처치 당시에는 재난에 대한 언급이 없었으나 환자의 이상반응 또는 이후 합병증의 원인이 된 외과적 및 기타 내과적 처치 (Y83-Y84)
질병이환과 사망의 외인의 후유증(Y85-Y89)
달리 분류된 질병이환 및 사망원인에 관련된 보조요인(Y90-Y98)

## 출력 조건
- 확장된 검색어는 최대 3개까지 반환합니다.
- 반드시 아래 JSON 형식으로만 출력하고, 그 외 텍스트를 포함하지 마.

## 출력 예시
1. 원본 질의: ESBL E.coli UTI
{
  "items": [
    {
      "q": "Extended spectrum betalactamase resistance Escherichia coli urinary tract infection",
      "categories": [
        "pathogen",
        "resistance"
      ]
    },
    {
      "q": "광범위 베타락탐계내성 대장균 요로감염 (ESBL E. coli UTI)",
      "categories": [
        "pathogen",
        "resistance"
      ]
    },
    {
      "q": "ESBL E.coli UTI",
      "categories": [
        "pathogen",
        "resistance"
      ]
    }
  ]
}

2. 원본 질의: 조영제에의해 발생한 아나필락시스쇼크
{
  "items": [
    {
      "q": "조영제에 의해 발생한 아나필락시스 쇼크",
      "categories": [
        "external"
      ]
    },
    {
      "q": "Contrast media–induced anaphylactic shock",
      "categories": [
        "external"
      ]
    },
    {
      "q": "조영제에의해 발생한 아나필락시스쇼크",
      "categories": [
        "external"
      ]
    }
  ],
  "external_matches": [
    "치료용으로 사용시 유해작용을 나타내는약물, 약제 및 생물학 물질(Y40-Y59)",
    "내과적 및 외과적 치료의 합병증(Y40-Y84)",
    "진단 및 치료용으로 사용시 유해사건과 관련된 의료장치(Y70-Y82)"
  ]
}
"""
        )
        prompt = f"원본 질의: {query}"

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "developer", "content": instruction},
                    {"role": "user", "content": prompt}
                    ],
                reasoning_effort="minimal",
                response_format={"type": "json_object"}
            )
            import json
            content = resp.choices[0].message.content
            data = json.loads(content)
            items = data.get("items", []) if isinstance(data, dict) else []
            expanded: List[str] = []
            classifications = {}
            external_matches: List[str] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                q = str(it.get("q", "")).strip()
                cats = [c for c in (it.get("categories", []) or []) if c in {"pathogen", "resistance", "external"}]
                if q:
                    expanded.append(q)
                    classifications[q] = cats
                if len(expanded) >= (max_expansions or 3):
                    break
            # Parse external_matches at top-level if provided by the model
            try:
                ext = data.get("external_matches", []) if isinstance(data, dict) else []
                if isinstance(ext, list):
                    for s in ext:
                        if isinstance(s, str) and s.strip():
                            external_matches.append(s.strip())
                    # Keep unique order, max 3
                    seen = {}
                    external_matches = [seen.setdefault(x, x) for x in external_matches if x not in seen][:3]
            except Exception:
                external_matches = []
            if not expanded:
                if self.require_llm:
                    raise RuntimeError("LLM returned no expansions.")
                expanded = [query]
                classifications = {query: []}
            self.logger.debug("expand_and_classify parsed %d items (external_matches=%d)", len(expanded), len(external_matches))
            out = {"expanded_queries": expanded, "classifications": classifications}
            if external_matches:
                out["external_matches"] = external_matches
            return out
        except Exception as e:
            self.logger.exception("expand_and_classify failed: %s", e)
            if self.require_llm:
                raise
            return {"expanded_queries": [query], "classifications": {query: []}}

    def classify_need(self, query: str) -> List[str]:
        """Return subset of ["pathogen","resistance","external"] using LLM if available."""
        if not self._client:
            if self.require_llm:
                raise RuntimeError("LLM required but not available. Set OPENAI_API_KEY.")
            return rule_based_classify(query)

        system = (
            "너는 분류기야. 입력 질의에 대해 다음 중 필요한 정보의 카테고리를 고르고, 반드시 JSON 배열로만 출력해.\n"
            "가능한 카테고리: pathogen, resistance, external\n"
            "출력 예시: [\"pathogen\", \"resistance\"]\n"
            "그 외 텍스트/설명 금지."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                reasoning_effort="minimal",
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            import json
            arr = json.loads(content)
            out = [c for c in arr if c in {"pathogen", "resistance", "external"}]
            if self.require_llm and not out:
                raise RuntimeError("LLM classification returned empty categories.")
            self.logger.debug("classify_need result=%s", out)
            return out
        except Exception as e:
            self.logger.warning("classify_need failed; falling back. error=%s", e)
            if self.require_llm:
                raise
            return rule_based_classify(query)


def rule_based_classify(text: str) -> List[str]:
    t = text.lower()
    cats = []
    if any(k in t for k in [
        "세균", "균", "감염", "병원체", "감염증",
        "pathogen", "infection", "bacteria", "virus", "박테리아", "바이러스",
        "균혈증", "패혈증", "결핵", "인플루엔자", "독감", "코로나", "covid"
    ]):
        cats.append("pathogen")
    if any(k in t for k in [
        "내성", "약제", "항생제", "resistance", "amr", "내성균", "감수성", "mic",
        "내성코드", "감염내성", "카바페넴", "메티실린", "mrsa", "vre", "esbl"
    ]):
        cats.append("resistance")
    if any(k in t for k in [
        # 일반 외인/사고
        "외인", "사고", "재해", "사망 원인", "external", "injury",
        # 낙상/추락/넘어짐 계열
        "낙상", "추락", "낙하", "넘어짐", "넘어져",
        # 교통사고/충돌
        "교통사고", "충돌", "접촉 사고", "차량", "자전거", "오토바이",
        # 상해/외상 표현
        "외상", "상해", "손상", "부상", "타박상", "열상", "절단", "자상", "찔림", "베임", "골절",
        # 물리/환경 요인
        "화상", "감전", "익사", "압궤", "끼임", "낙석", "붕괴",
        # 독성/중독
        "중독", "독성", "독극물", "toxic",
        # 폭력/의도성
        "폭행", "폭력", "자살", "자해"
    ]):
        cats.append("external")
    return list(dict.fromkeys(cats))  # preserve order, unique
