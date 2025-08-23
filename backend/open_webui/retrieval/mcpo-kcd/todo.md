open-webui의 mcpo(https://github.com/open-webui/mcpo) 를 사용해서 파이썬기반 langgraph(openai의 gpt-5-mini기반)를 활용한 kcd mcpo 서버를 만들꺼야

역할 : 이 kcd mcpo 서버는 쿼리를 받아서 그 쿼리의 내용이 "다른 장에서 분류된 질환의 원인으로서의 세균 정보(data\kcd_kb_pathogen.txt)" 가 필요한지, "약제내성코드에대한 추가 정보(data\kcd_kb_resistance.txt)" 가 필요한지, "질병의 외인 관련 추가 참고 코드 정보(data\kcd_kb_external.txt)"가 필요한지 판단해서 분류하고 임베딩 검색을 수행해서 관련 정보를 제공하는 mcpo 서버야

1. 사용자의 쿼리를 받아오면 그 쿼리에 약어가 포함되어 있는지 확인한 후 약어가 포함되어 있다면 LLM 기반 쿼리 확장(최대 3개)를 수행
2. 이 확장된 쿼리별로 LLM이 추가로 그 쿼리가 "다른 장에서 분류된 질환의 원인으로서의 세균 정보(data\kcd_kb_pathogen.txt)" 가 필요한지, "약제내성코드에대한 추가 정보(data\kcd_kb_resistance.txt)" 가 필요한지, "질병의 외인 관련 추가 참고 코드 정보(data\kcd_kb_external.txt)"가 필요한지 판단하고 분류에 맞게 파일을 참고해서 sentence_transformers 라이브러리를 사용해서 허깅페이스의 dragonkue/multilingual-e5-small-ko-v2 모델을 사용하여 임베딩 검색 수행

