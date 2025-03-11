# **Dataset Info.**

## 출저: [Dacon](https://dacon.io/competitions/official/236455/overview/rules)

![image](https://github.com/user-attachments/assets/d695cfd1-69e8-4f4f-9a13-f1fb9d5ed526)

※ S-Bert Cosine 유사도는 예측(생성) 문장과 정답 문장간의 의미론적 유사성을 측정합니다. 

※ S-Bert Cosine 유사도에 활용되는 정답(GT)의 Embedding Vector는 ‘jhgan/ko-sbert-sts’ 모델을 통해 추출되었습니다.

※ Jaccard 유사도는 예측(생성) 문장과 정답 문장에서 사용된 단어들의 집합을 비교하여 어휘적 유사성을 측정합니다.

※ S-Bert Cosine 유사도에 활용되는 Embedding Vector는 반드시 Jaccard 유사도에 활용되는 Text이어야 합니다.
    
```
def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0


def jaccard_similarity(text1, text2):
    """자카드 유사도 계산"""
    set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
    intersection = len(set1.intersection(set2))  # 교집합 크기
    union = len(set1.union(set2))  # 합집합 크기
    return intersection / union if union != 0 else 0
```

**train.csv [파일] - 학습 가능**
- ID : 샘플별 고유 ID
- 발생일시
- 사고인지 시간
- 날씨
- 기온
- 습도
- 공사종류
- 연면적
- 층 정보
- 인적사고
- 물적사고
- 공종
- 사고객체
- 작업프로세스
- 장소
- 부위
- 사고원인
- 재발방지대책 및 향후조치계획

**건설안전지침 [폴더] - 학습 가능**
- 104개의 건설안전지침 PDF

**test.csv [파일]**
- ID : 샘플별 고유 ID
- 발생일시
- 사고인지 시간
- 날씨
- 기온
- 습도
- 공사종류
- 연면적
- 층 정보
- 인적사고
- 물적사고
- 공종
- 사고객체
- 작업프로세스
- 장소
- 부위
- 사고원인

**sample_submission.csv [파일] - 제출 양식**
- ID : 샘플별 고유 ID
- 재발방지대책 및 향후조치계획 : 생성된 답변
- vec_0, vec_1 ... vec_767 : 생성된 답변을 768 차원의 Embedding Vector로 표현된 결과
