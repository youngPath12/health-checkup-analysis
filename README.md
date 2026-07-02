# 건강검진 데이터 분석 실습 (health-checkup-analysis)

AI·의료바이오데이터 융합인재 개발 과정 — 통계·데이터 분석 실습용 저장소입니다.
건강검진 데이터(400명 × 14변수)를 사용해 **데이터 진단부터 대사증후군 예측 모델까지**
전 과정을 하나의 노트북으로 다룹니다.

## 분석 개요

검진 데이터를 받아 전 변수를 체계적으로 탐색하고, 대사증후군(metabolic_syndrome)을
결과변수로 한 위험요인 분석·예측까지 수행합니다.

| STEP | 내용 |
|------|------|
| STEP 0 | 환경 준비 & 데이터 불러오기 (한글 폰트 설정, Drive 연결) |
| STEP 1 | 데이터 진단 — 구조·타입·결측 확인 및 중앙값 대치 |
| STEP 2 | 기술통계 — 중심·산포·분포, 변동계수(CV), 범주형 빈도 |
| STEP 3 | 연속형 검정 — 정규성·등분산 점검 후 t검정/ANOVA 자동 선택 + 사후검정 |
| STEP 4 | 범주형 검정 — 기대빈도 점검 후 카이제곱/Fisher 자동 선택 + Cramér's V |
| STEP 5 | 상관·회귀 — 상관행렬, VIF, 로지스틱 회귀, 조정 오즈비, ROC/AUC |

## 데이터 구조 (14변수)

- **연속형 (8):** age, bmi, waist, sbp, glucose, tg, hdl, crp
- **범주형 (5):** sex, region, smoking, exercise_freq, metabolic_syndrome
- **관측치:** 400행

> 변수 정의·코딩 상세는 `data/README.md` 참고.

## 실행 방법

1. Google Colab에서 `notebooks/analysis.ipynb`를 엽니다.
   (GitHub 노트북 URL의 `github.com`을 `githubtocolab.com`으로 바꾸면 바로 열립니다.)
2. 첫 셀에서 패키지를 설치합니다.
```
   !pip install -r requirements.txt
```
3. 데이터를 연결합니다. 두 가지 방식 중 택1:
   - **업로드 방식:** STEP 0의 `files.upload()` 셀 실행 → `건강검진_실습데이터.xlsx` 선택
   - **드라이브 방식:** Drive 마운트 후 `MyDrive/강의자료/실습데이터/` 경로에서 읽기
4. `런타임 > 모두 실행`으로 처음부터 끝까지 실행합니다.

## ⚠️ 데이터 관리 원칙 (중요)

- **원자료(환자 검진 데이터)는 이 저장소에 포함하지 않습니다.**
  민감 정보 유출을 막기 위해, 공개 저장소에는 원자료를 절대 올리지 않습니다.
- 원자료는 **Google Drive** 등 접근이 통제된 저장소에서 관리합니다.
- GitHub에는 **실행 코드, 데이터 설명서, README, 분석 절차 문서**만 둡니다.
- 결과 파일(그림·표)도 필요 시 Drive에 보관하고, 저장소에는 생성 코드만 남깁니다.

## 저장소 구조

```
health-checkup-analysis/
├── README.md              # 본 문서
├── requirements.txt       # 분석 환경 (패키지 목록)
├── notebooks/
│   └── analysis.ipynb     # 전체 분석 노트북
├── utils/
│   └── utils.py           # 분석 관련 함수 정의
└── data/
    └── README.md          # 데이터 설명서 (원자료는 미포함)
```

## 작성자

Kyeong Taek Oh, Ph.D. — Department of Biomedical Engineering, Yonsei University College of Medicine
Joy Rogers, Ph. D. - 
