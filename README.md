**RAG(Retrieval-Augmented Generation)**

## 주제
**이건설공사 사고 상황 데이터를 바탕으로 사고 원인을 분석하고 재발방지 대책을 포함한 대응책을 자동으로 생성하는 RAG 모델을 개발**

## 배경
이 프로젝트에서는 대규모 언어 모델(LLM)을 사용하는 대신, RAG(Retrieval-Augmented Generation) 방식을 선택한 이유는 환경과 효율성을 고려한 최적화 때문입니다.
대규모 언어 모델을 사용하려면 높은 연산 자원과 메모리 요구 사항을 충족해야 하며, 이는 실제 프로덕션 환경에서 비용 및 성능 면에서 부담이 될 수 있습니다. 특히, 사고 분석과 재발 방지 대책을 생성하는 작업은 정확하고 실용적인 정보를 추출해야 하므로, 단순한 LLM을 사용하는 방식보다는 내부 데이터베이스에서 관련 정보를 빠르게 검색하고, 이를 바탕으로 구체적인 대응책을 생성하는 것이 더욱 효율입니다.

## 학습 데이터 전처리 프로세스 :

**1)종류 분할**
공사 종류에 대한 정보를 대분류, 중분류, 소분류로 세분화하여 구조화합니다.

예시: 건축 / 건축물 / 근린생활시설
- 공사종류(대분류) = 건축
- 공사종류(중분류) = 건축물
- 공사종류(소분류) = 근린생활시설

사고 객체 역시 동일하게 **사고 객체(대분류), 사고 객체(중분류), 사고 객체(소분류)**로 분할하여 보다 정교한 분류를 진행합니다. 이 과정을 통해 각 사고 데이터를 보다 정확하고 체계적으로 분류하여, 나중에 사고 원인 분석과 대책을 보다 정교하게 도출할 수 있습니다.

## PDF 문서 전처리 프로세스 :

1) Load

PDF 문서를 로드하는 과정에서 fitz를 사용하면 페이지 번호만 제공되며, langchain.PdfReader는 한글 인코딩 처리에 우수한 성능을 보이고, 속도 면에서도 안정적입니다.
따라서 PDF 문서 로딩을 위해 langchain.PdfReader를 사용하여 문서를 처리합니다. 이때, 불필요한 요소들이 포함된 문서도 존재하므로 이를 처리하는 과정이 중요합니다.

PDF 문서에는 검색에 Unnessesary한 사진, 특수 문자, 공백, 줄바꿈, "KOSHAGUIDE C552015" 부록을 포함하는 문구 등 기타 unnecessary한 요소들을 처리.

2) 문서 정리 및 클린징
PDF 문서에는 검색에 불필요한 사진, 특수 문자, 공백, 줄바꿈, 부록 등의 불필요한 요소들이 포함될 수 있습니다.
예시: "KOSHAGUIDE C552015" 부록 문구 등은 Unnecessary한 데이터로, 이를 제거하여 실제 사고 원인 분석에 유용한 텍스트만 남길 수 있도록 클린징 작업을 진행합니다.
   
4) Split
RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할합니다.
이 방법은 문서를 일정 길이로 자르고, 각 조각을 개별적으로 다룰 수 있게 해주며, 이후 모델 학습에 최적화된 텍스트를 제공합니다.
이를 통해 사고 원인 및 관련 문서들을 더 잘게 나누어 처리할 수 있습니다.

5) Embed
텍스트 임베딩에는 SentenceTransformer("all-MiniLM-L6-v2") 모델을 사용합니다. 이 모델은 빠르고 효율적으로 텍스트를 벡터 형태로 변환하여 의미 기반 검색에 적합한 표현을 제공합니다.
이 과정을 통해 각 문서가 의미적으로 잘 표현된 벡터로 변환되어, 검색 및 후속 처리에서 높은 성능을 발휘합니다.



7) Store
문서를 하이브리드 검색 방식을 통해 저장합니다.
BM25 Retriever는 키워드 기반 검색을, FAISS Retriever는 의미 기반 검색을 담당합니다.
이 두 방식을 하이브리드 방식으로 결합하여 정확한 사고 원인 문서를 검색합니다.

BM25는 간단한 키워드 검색을 통해 빠르게 관련성 있는 문서를 추출합니다.
FAISS는 의미적 유사성을 기반으로 보다 정교한 검색을 수행하여, 사고 원인과 밀접하게 관련된 문서를 찾습니다.
이 두 가지 검색 방법을 결합하여 정확하고 효율적인 사고 원인 관련 문서를 검색할 수 있습니다.
또한, 특정 스코어 임계치(Threshold) 이상에 해당하는 문서만 저장하여, 불필요한 문서를 필터링하고, 관련성이 높은 문서들만 저장합니다. 이를 통해 최적화된 문서 저장이 가능하며, 사고 원인 분석의 품질을 향상시킬 수 있습니다.

<img width="1358" alt="image" src="https://github.com/user-attachments/assets/a7b96b20-af95-4ebe-a044-0dc6fa03c2c7" />
