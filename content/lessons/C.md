---
id: C
part: 9
title: 수식·기호표
chapters: []
version: 1
updated: 2026-10-02
status: draft
---

# 부록 C 수식·기호표

## 이 부록의 쓰임

1~61강 원고에 나온 수식은 1,650개이지만 대부분은 계산 예(숫자를 넣은 식)이거나 기호 하나임. 이 부록은 그중 각 강의 핵심 식 120여 개만 골라 부별 표로 모으고, 책 전체에서 쓰는 기호를 한 표로 정리했음. 새 내용은 없고, 강 본문을 다시 찾아가게 돕는 색인임. 식의 뜻과 계산 예는 괄호 안 강에서 읽음.

찾는 법은 세 가지임.
- **식 이름을 알 때**: 그 식이 속한 부의 표에서 이름을 찾음. 표의 "강"이 처음 나온 강이고, 둘 이상이면 뒤 강에서 다시 쓰거나 고쳐 쓴 것임
- **기호만 보일 때**: 아래 기호 관례 표에서 기호를 찾고, 첫 등장 강과 주의 칸을 봄
- **지시문에 식 이름이 나올 때**: 부록 A로 지시를 해석하고, 이 부록에서 식을 확인함

같은 기호가 강마다 다른 뜻으로 쓰인 경우가 많음. 책 한 권이 통계·영상 처리·딥러닝·Diagnostics를 다 다루고, 각 분야의 관례 기호를 그대로 따랐기 때문임. 예를 들어 $\sigma$는 1부에서 표준편차, 12강에서 괄호를 붙인 $\sigma(z)$는 시그모이드, 56강에서는 특잇값임. $p$는 확률·p값·계수 개수·화소 크기·드롭아웃 비율로 다섯 번 넘게 뜻이 바뀜. 기호를 보면 **그 강 안에서 정의한 뜻이 우선**이고, 앞 강에서 본 뜻을 그대로 가져오면 틀리기 쉬움. 기호 관례 표의 "주의" 칸이 이 겹침을 모아 둔 곳임.

"첫 등장 강"은 원고에서 수식을 모두 뽑아 찾는 보조 스크립트로 확인한 값임. 원고 문장에 이름만 먼저 나오고 수식은 뒤 강에 나오는 경우가 있어, 그 개념을 처음 설명한 강과 다를 수 있음(예: IoU는 35강에서 설명하지만 수식으로는 34강의 PQ 식에 먼저 나옴).

## 기호 관례

책 전체에서 쓰는 기호와 표기 습관임. "첫 등장 강"은 그 기호가 수식에 처음 나온 강이고, "주의"는 다른 강에서 다른 뜻으로 쓴 경우임.

| 기호 | 뜻 | 첫 등장 강 | 주의(다른 뜻) |
|---|---|---|---|
| $n$ | 표본(관측치) 수 | 1 | 2강 이항분포에서는 시행 횟수, 20강 $2^n$에서는 비트 수 |
| $N$ | 대문자 개수(전체·화소·토큰 등) | 7 | 7강 동별 화소 수 $N_i$, 24강 RANSAC 반복 수, 30강 입력 한 변 크기, 32·56강 토큰 수, 44강 북향 좌표, 58강 센서 전체 집합, 61강 $N_{10}$ 같은 표의 칸 수 |
| $x_i$, $y_i$ | $i$번째 관측의 값(설명변수·반응변수) | $x_i$ 1, $y_i$ 5 | 대문자 $X$는 2강 확률변수, 7·8강 설계 행렬, 26강 세계 좌표, 57·59강 입력 영상, 60강 활성 행렬 |
| $\bar{x}$ | 표본평균(막대 = 평균) | 1 | 55강 $\bar{\mathbf{p}}$는 여러 번 낸 확률 벡터의 평균 |
| 모자 $\hat{x}$ | 추정값·예측값($\hat{p}$, $\hat{\theta}$, $\hat{y}$) | $\hat{p}$·$\hat{\theta}$ 3, $\hat{y}$ 6 | 22강 $\hat{x}$는 Lee 필터 결과, 29강 정규화한 값, 42강 칼만 예측, 50·60강 역양자화한 값, 28강 $\hat{m}$·$\hat{s}$는 편향 보정값 |
| $e_i$ | 잔차 $y_i - \hat{y}_i$ | 6 | 오차항 $\varepsilon_i$(관측 못 하는 참 오차)와 구분 |
| $s$ | 표본 표준편차 | 1 | 7강 잔차 표준편차, 19강 실루엣 $s_i$, 24강 RANSAC 최소 표본 수, 26강 축 기울어짐, 28강 Adam의 기울기 제곱 이동 평균, 36강 스트라이드·점수, 50·60강 양자화 축척, 53강 헤드 로짓 $s_k$ |
| $\mu$, $\sigma$ | 모집단 평균·표준편차 | 2 | $\mu$: 19강 군집 중심, 28강 모멘텀 계수, 29강 배치 평균 $\mu_B$, 56강 채널 평균 활성, 57강 표적·배경 평균. $\sigma$: 22강 가우시안 폭, 37강 Soft-NMS 폭, 56강 특잇값 $\sigma_k$ |
| $\sigma(\cdot)$ | 시그모이드 함수 | 12 | 괄호와 인수가 붙으면 함수, 없으면 표준편차. 28강 $\sigma(z)(1-\sigma(z))$도 시그모이드 |
| $\sigma^2$, $s^2$ | 모분산, 표본분산 | $s^2$ 1, $\sigma^2$ 2 | 23강 $\sigma_B^2$는 클래스 간 분산, 29강 $\sigma_B^2$는 배치 분산 |
| $p$ | 확률 | 2 | 4강 p값, 6~8강 절편을 포함한 계수 개수, 20·26강 센서 화소 크기, 29강 드롭아웃 비율, 32강 토큰 위치, 39강 정밀도 $p(r)$ |
| $\alpha$ | 유의수준 | 4 | 11강 엘라스틱넷 섞음 비율, 16강 가지치기 강도, 18강 포컬 손실 가중, 27강 Leaky ReLU 기울기, 35강 CIoU 가중, 36강 TAL 지수, 42강 HOTA의 IoU 기준, 45강 베타 분포 모수, 60강 SmoothQuant 이동 강도 |
| $\beta$ | 2종 오류 확률($1-\beta$ = 검정력) | 4 | 6·10·12강 회귀계수(모형의 참값 $\beta$, 추정값 $b$), 28강 Adam 감쇠율 $\beta_1$·$\beta_2$, 29강 배치 정규화 이동, 36강 TAL 지수, 57강 산란 계수 |
| $\lambda$ | 포아송 분포의 평균 | 2 | 10강 박스-콕스 지수, 11·14강 규제 강도, 18강 SMOTE 보간 비율, 24강 고윳값, 26강 경도, 28·29강 가중치 감쇠 계수, 36강 비용 가중, 45강 믹스업 비율, 59강 IRM 벌점 |
| $\theta$ | 추정할 모수 | 3 | 24강 기울기 방향, 25·26강 지점에서 본 시선이 연직에서 기운 각, 57강 오프나디르 각, 28·46강 모델 파라미터, 41강 회전 박스 각도 |
| $\eta$ | 학습률 | 28 | 46강 $\eta_t$는 걸음 $t$의 학습률(스케줄) |
| $\varepsilon$, $\epsilon$ | $\varepsilon$: 오차항·잡음 | $\varepsilon$ 6, $\epsilon$ 28 | $\epsilon$은 28·29강에서 분모가 0이 되지 않게 더하는 작은 수, 45강 잡음, 56강 비활성 문턱. $\varepsilon$은 19강 DBSCAN 반경, 29강 라벨 스무딩 비율, 44강 라벨 노이즈 비율 |
| $\gamma$ | 보조 회귀의 계수 | 7 | 18강 포컬 손실 집중 지수, 21강 감마, 29강 배치 정규화 배율 |
| $\rho$ | 스피어만 순위상관 | 5 | 8강 후버 손실 함수 $\rho(e)$, 9강 자기상관 계수, 16강 트리 간 상관, 20·45강 반사율, 35강 중심 거리 |
| $r$ | 피어슨 상관계수 | 5 | 7·8강 스튜던트화 잔차 $r_i$, 9강 잔차의 1차 자기상관, 22·50강 행 번호, 24강 비율 검정 문턱, 26강 반경, 36강 상자 오른쪽 거리, 39강 재현율 |
| $\kappa$ | 카파(우연 일치를 뺀 일치도) | 17 | - |
| $k$, $K$ | $k$: 개수·색인(2강 성공 횟수) | $k$ 2, $K$ 12 | $k$: 15강 겹 수, 19강 군집 수, 22강 커널 크기·샤프닝 강도, 24강 해리스 상수, 54강 커널 함수. $K$: 12·27·56강 클래스 수, 22강 커널, 26강 내부 파라미터 행렬, 30강 커널 크기, 32강 키 행렬, 42강 칼만 이득, 53강 상위 상자 수 |
| $z$ | 표준화 점수 | 1 | 12·27강 선형 결합(로짓), 22강 SAR 관측값, 42강 관측(탐지), 50·60강 영점, 55강 로짓 $z_k$ |
| $t$, $T$ | t 통계량($T$는 그 확률변수) | 4 | $t$: 9·42강 시점, 23강 문턱, 26강 이동 벡터, 36강 TAL 정렬 점수, 46강 걸음 수, 52강 IoU 임계값, 57강 투과율. $T$: 27·55강 온도, 46강 전체 걸음 수, 60강 메모리 전송량 |
| $w$, $W$ | 가중치(5강 공간 가중 $w_{ij}$, 그 합 $W$) | 5 | $W$: 19강 군집 내 제곱합, 22강 Lee 가중, 27·32·60강 가중치 행렬, 41·44·45강 너비, 54·61강 바서슈타인 거리 $W_1$. $w$: 24강 인라이어 비율, 35강 상자 너비 |
| $b$ | 회귀계수 추정값 $b_0$, $b_1$ | $b_0$ 6 | 27강 신경망의 편향 $b$, 45강 밴드 번호, 54강 배경 클래스 |
| $H$ | 가설 $H_0$, $H_1$ | 4 | 8강 햇 행렬, 16·55강 엔트로피, 20·26강 촬영 고도, 25강 호모그래피, 41·44강 높이 |
| $A$, $B$ | 사건 $A$, $B$ | 2 | 23강 영상과 구조 요소, 35강 IoU의 두 상자. $B$는 3강 부트스트랩 반복 수, 16강 모델 수, 26강 기선, 53강 경계 띠 $B(Y_c)$, 60강 메모리 대역폭. $A$는 57강 정확도·대기광 |
| 예측·정답 상자 | 정답에 위첨자 $gt$ 또는 별표 | $gt$ 35 | 52강은 정답 상자를 $w^*$, $h^*$처럼 별표로, 59강은 예측 $b$, 정답 $B_i$로 씀 |
| $G$ | 지니 불순도 | 16 | 21강 녹색, 22강 가우시안 커널, 24강 기울기 $G_x$, 54강 지니 계수, 61강 게이트 |
| $d$ | 효과크기 $d_z$ | 4 | 5강 순위 차, 9강 더빈-왓슨 통계량, 13강 트리 깊이, 26강 시차, 30강 팽창 계수, 32강 키 차원, 40강 경계 허용 거리, 42강 마할라노비스 거리, 57강 빛이 지나는 거리 |
| $c$, $C$ | $c$: 후버 상수 | $c$ 8, $C$ 19 | $c$: 28강 클리핑 상한, 35강 외접 상자 대각선, 53·54강 클래스. $C$: 19강 군집, 30·56강 채널 수, 35강 외접 상자, 54강 클래스 수, 59강 맥락 변수 |
| $\ln$, $\log$ | $\ln$은 자연로그 | $\ln$ 7, $\log$ 16 | 밑을 적은 것은 16강 $\log_2$(엔트로피), 60강 $\log_{10}$(dB). 밑 없는 $\log$는 24강(분자·분모가 같은 밑이라 밑과 무관)과 36강(도구 구현은 보통 자연로그)뿐 |
| $e^{x}$, $\exp$ | 지수함수 | 2 | 두 표기는 같은 뜻. 식이 길면 $\exp(\cdot)$로 씀 |
| $E[\cdot]$, $\mathbb{E}[\cdot]$ | 기댓값 | $E$ 2, $\mathbb{E}$ 14 | 같은 뜻의 두 표기. 52강 $E_{\text{cls}}$ 등은 오차 유형별 개수, 53강 $E_c$는 틀린 화소, 59강 $E$는 환경 변수 |
| $\mathrm{Var}$, $\mathrm{Cov}$ | 분산, 공분산 | $\mathrm{Var}$ 2, $\mathrm{Cov}$ 5 | - |
| 굵은 글자 $\mathbf{x}$ | 벡터임을 강조할 때 | 31 | 대부분 강은 굵게 쓰지 않음. 소문자는 스칼라·벡터, 대문자는 행렬($X$, $H$, $W$, $Q$·$K$·$V$)로 읽음 |
| $^\top$ | 전치(행과 열을 바꿈) | 7 | - |
| $\lvert \cdot \rvert$ | 절댓값 | 1 | 35·53강 영역의 넓이(화소 수), 58강 집합의 원소 수 |
| $\lVert \cdot \rVert$ | 노름(벡터 길이) | 19 | - |
| $\cap$, $\cup$ | 교집합, 합집합 | 2 | 35강부터 상자 넓이의 겹침·합침 |
| $\mathbb{I}$, $\mathbf{1}$ | 지시 함수(참이면 1) | 56 | 같은 뜻의 두 표기 |
| $\oplus$, $\ominus$ | 팽창, 침식 | 23 | 53강 경계 띠 정의에 다시 씀 |
| $\leftarrow$ | 갱신(오른쪽 값을 왼쪽에 넣음) | 28 | 59강 인과 그래프에서는 화살표 방향 |
| $\lfloor \cdot \rfloor$ | 내림 | 30 | - |
| $\vert$, $\mid$ | 조건부 확률의 "주어졌을 때" | 4 | 26강 $[R \mid t]$는 행렬을 옆으로 이어 붙임 |
| $\mathrm{TP}$·$\mathrm{FP}$·$\mathrm{FN}$·$\mathrm{TN}$ | 혼동행렬의 네 칸 | 17 | 탐지에서는 TN을 세지 않음(38강) |
| $\mathrm{IoU}$ | 겹침 넓이 ÷ 합친 넓이 | 34 | 설명은 35강. 42강 HOTA·59강 CHR에서 짝짓기 기준으로 다시 씀 |
| $p_k$ (소프트맥스) | 로짓 $z_k$를 합이 1인 확률로 | 12 | 32강은 $\mathrm{softmax}(\cdot)$로 씀. 53강은 헤드 로짓을 $s_k$로 쓰고, 56강은 특잇값 비율을 $p_k$라 부름 |
| $\mathrm{do}(\cdot)$ | 개입(바깥에서 값을 정함) | 59 | - |

## 1부 데이터를 숫자로 읽기

| 이름 | 식 | 강 |
|---|---|---|
| 표본평균 | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | 1 |
| 표본분산·표준편차 | $s^2 = \frac{1}{n-1}\sum_{i}(x_i - \bar{x})^2,\ s = \sqrt{s^2}$ | 1 |
| 사분위범위 | $\mathrm{IQR} = Q_3 - Q_1$ | 1 |
| 중앙값 절대편차 | $\mathrm{MAD} = \operatorname{median}(\lvert x_i - \operatorname{median}(x) \rvert)$ | 1 |
| z점수 | $z_i = \frac{x_i - \bar{x}}{s}$ | 1 |
| 이항분포 | $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$ | 2 |
| 포아송 분포 | $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$ | 2 |
| 정규분포 밀도 | $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ | 2 |
| 표준오차(평균·비율) | $\mathrm{SE}(\bar{x}) = \frac{s}{\sqrt{n}},\ \mathrm{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ | 3 |
| 95% 신뢰구간 | $\hat{\theta} \pm 1.96 \times \mathrm{SE}$ | 3 |
| p값 | $p = P(\lvert T \rvert \ge \lvert t_{\text{관측}} \rvert \mid H_0)$ | 4 |
| BH 절차 | $p_{(k)} \le \frac{k}{m}\,q$ | 4 |
| 피어슨 상관계수 | $r = \frac{\mathrm{Cov}(x, y)}{s_x\, s_y}$ | 5 |
| 모란 지수 | $I = \frac{n}{W}\cdot\frac{\sum_i\sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$ | 5 |

## 2부 관계를 모형으로

| 이름 | 식 | 강 |
|---|---|---|
| 단순회귀 모형 | $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$ | 6 |
| 최소제곱 기울기·절편 | $b_1 = r\,\frac{s_y}{s_x},\ b_0 = \bar{y} - b_1 \bar{x}$ | 6 |
| 결정계수 | $R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}$ | 6 |
| 스튜던트화 잔차 | $r_i = \frac{e_i}{s\sqrt{1-h_{ii}}},\ t_i = \frac{e_i}{s_{(i)}\sqrt{1-h_{ii}}}$ | 7 |
| 분산팽창계수 | $\mathrm{VIF}_j = \frac{1}{1 - R_j^2}$ | 8 |
| 햇 행렬 | $\hat{y} = Hy,\ H = X(X^\top X)^{-1}X^\top$ | 8 |
| 쿡의 거리 | $D_i = \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1 - h_{ii}}$ | 8 |
| 더빈-왓슨 통계량 | $d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2} \approx 2(1 - r)$ | 9 |
| 박스-콕스 변환 | $y^{(\lambda)} = \frac{y^{\lambda} - 1}{\lambda}$ ($\lambda \ne 0$), $\ln y$ ($\lambda = 0$) | 10 |
| 규제 회귀(벌점은 릿지 $\sum_j \beta_j^2$, 라쏘 $\sum_j \lvert \beta_j \rvert$) | $\min_{\beta} \sum_i (y_i - \hat{y}_i)^2 + \lambda P(\beta)$ | 11 |
| AIC·BIC | $-2\ln L + 2k$, $-2\ln L + k\ln n$ | 11 |
| 로짓 | $\operatorname{logit}(p) = \ln\frac{p}{1-p}$ | 12 |
| 시그모이드 | $p = \sigma(z) = \frac{1}{1 + e^{-z}}$ | 12 |
| 로그 우도(이진) | $\ln L = \sum_i [y_i \ln p_i + (1 - y_i)\ln(1 - p_i)]$ | 12 |
| 오즈비 | $\mathrm{OR} = e^{\beta_1}$ | 12 |
| 소프트맥스(다항 로지스틱) | $p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$ | 12, 27 |

## 3부 머신러닝의 기본기

| 이름 | 식 | 강 |
|---|---|---|
| 편향-분산 분해 | $\mathbb{E}[(y - \hat{f}(x))^2] = \text{편향}^2 + \text{분산} + \sigma^2$ | 14 |
| 지니 불순도·엔트로피 | $G = 1 - \sum_k p_k^2,\ H = -\sum_k p_k \log_2 p_k$ | 16 |
| 배깅 예측의 분산 | $\rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2$ | 16 |
| 정밀도·재현율 | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$, $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ | 17 |
| F1 | $F_1 = \frac{2 \times \text{정밀도} \times \text{재현율}}{\text{정밀도} + \text{재현율}}$ | 17 |
| 카파 | $\kappa = \frac{p_o - p_e}{1 - p_e}$ | 17 |
| 포컬 손실 | $\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma}\ln p_t$ | 18 |
| SMOTE 합성 표본 | $x_{\text{new}} = x_i + \lambda\,(x_j - x_i)$ | 18 |
| K-평균 목적(군집 내 제곱합) | $W = \sum_{j=1}^{k} \sum_{i \in C_j} \lVert x_i - \mu_j \rVert^2$ | 19 |
| 실루엣 | $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$ | 19 |

## 4부 영상과 기하

| 이름 | 식 | 강 |
|---|---|---|
| 지상 표본 거리 | $\mathrm{GSD} = \frac{H \times p}{f}$ | 20 |
| DN → 반사율 | $\rho = \mathrm{DN} \times \text{스케일} + \text{오프셋}$ | 20 |
| NDVI | $(\text{근적외} - \text{적색}) / (\text{근적외} + \text{적색})$ | 20 |
| 선형 스트레칭 | $v' = 255 \times \frac{v - a}{b - a}$ | 21 |
| 감마 보정 | $v_{\text{out}} = v_{\text{in}}^{1/\gamma}$ | 21 |
| 합성곱(3×3) | $g(r, c) = \sum_{i}\sum_{j} K(i, j)\, f(r - i, c - j)$ | 22 |
| 가우시안 커널 | $G(i, j) \propto \exp\!\left(-\frac{i^2 + j^2}{2\sigma^2}\right)$ | 22 |
| Lee 필터 | $\hat{x} = m + W(z - m),\ W = 1 - C_u^2/C_i^2$ | 22 |
| 오츠 클래스 간 분산 | $\sigma_B^2(t) = w_0 w_1 (\mu_0 - \mu_1)^2$ | 23 |
| 열림·닫힘 | $A \circ B = (A \ominus B) \oplus B,\ A \bullet B = (A \oplus B) \ominus B$ | 23 |
| 해리스 응답 | $R = \det M - k\,(\operatorname{tr} M)^2$ | 24 |
| RANSAC 반복 수 | $N = \frac{\log(1 - p)}{\log(1 - w^s)}$ | 24 |
| 호모그래피 | $(u, v, w)^\top = H(x, y, 1)^\top,\ x' = u/w,\ y' = v/w$ | 25 |
| 기복 변위 | $\Delta \approx h \tan\theta$ | 25 |
| 카메라 투영 | $Z_c (u, v, 1)^\top = K [R \mid t] (X_w, Y_w, Z_w, 1)^\top$ | 26 |
| 스테레오 깊이 | $Z = \frac{fB}{d},\ \delta Z \approx \frac{Z^2}{fB}\,\delta d$ | 26 |

## 5부 딥러닝

| 이름 | 식 | 강 |
|---|---|---|
| 퍼셉트론 | $z = w_1x_1 + \cdots + w_nx_n + b$, $z > 0$이면 1 | 27 |
| ReLU·Leaky ReLU·GELU | $\max(0, z)$, 음수 쪽 $\alpha z$, $z \cdot \Phi(z)$ | 27 |
| 교차 엔트로피(한 표본) | $L = -\ln p_{\text{정답}}$ | 12, 18, 28 |
| 경사하강(SGD) | $\theta \leftarrow \theta - \eta\, g$ | 28 |
| 모멘텀 | $v \leftarrow \mu v + g,\ \theta \leftarrow \theta - \eta\, v$ | 28 |
| Adam 이동 평균 | $m \leftarrow \beta_1 m + (1-\beta_1) g,\ s \leftarrow \beta_2 s + (1-\beta_2) g^2$ | 28 |
| Adam 갱신 | $\theta \leftarrow \theta - \eta\, \hat{m}/(\sqrt{\hat{s}} + \epsilon)$ | 28 |
| 가중치 감쇠(AdamW) | $\theta \leftarrow \theta - \eta\lambda\theta$ | 28, 29 |
| 기울기 클리핑 | $g \leftarrow g \cdot \min(1,\ c/\lVert g \rVert)$ | 28 |
| 라벨 스무딩 | $y_k^{\text{LS}} = (1-\varepsilon)\,y_k + \varepsilon/K$ | 29 |
| 배치 정규화 | $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\ y = \gamma\hat{x} + \beta$ | 29 |
| 합성곱 층 파라미터 수 | $K \times K \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$ | 30 |
| 출력 크기 | $N_{\text{out}} = \lfloor (N + 2P - K)/S \rfloor + 1$ | 30 |
| 수용영역 | $R_l = R_{l-1} + (K_l - 1) \times \prod_{i<l} S_i$ | 30 |
| 잔차 연결 | $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ | 31 |
| 스케일드 내적 어텐션 | $\mathrm{softmax}(QK^{\top}/\sqrt{d})\, V$ | 32 |

## 6부 태스크와 평가

| 이름 | 식 | 강 |
|---|---|---|
| 파놉틱 품질 | $\mathrm{PQ} = \frac{\sum_{\text{짝}} \mathrm{IoU}}{\mathrm{TP} + \frac{1}{2}\mathrm{FP} + \frac{1}{2}\mathrm{FN}}$ | 34 |
| IoU | $\frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert} = \frac{\lvert A \cap B \rvert}{\lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert}$ | 35 |
| GIoU | $\mathrm{IoU} - \frac{\lvert C \rvert - \lvert A \cup B \rvert}{\lvert C \rvert}$ | 35 |
| DIoU | $\mathrm{IoU} - \rho^2/c^2$ ($\rho$: 중심 거리, $c$: 외접 상자 대각선) | 35 |
| CIoU | $\mathrm{DIoU} - \alpha v,\ \alpha = \frac{v}{(1-\mathrm{IoU}) + v}$ ($v$: 종횡비 차이 항) | 35 |
| 앵커 상자 부호화 | $t_x = \frac{x - x_a}{w_a},\ t_w = \log\frac{w}{w_a}$ ($t_y$, $t_h$도 같은 꼴) | 36 |
| TAL 정렬 점수 | $t = s^{\alpha} u^{\beta}$ | 36 |
| Soft-NMS(가우시안) | $s_i \leftarrow s_i \, e^{-\mathrm{IoU}^2/\sigma}$ | 37 |
| AP 정밀도 보간 | $p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$ | 39 |
| COCO 크기 구간(넓이) | 소형 $< 32^2$, 중형 $32^2$~$96^2$, 대형 $\ge 96^2$ | 39 |
| 클래스 IoU(분할) | $\mathrm{IoU}_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FP}_k + \mathrm{FN}_k}$ | 40 |
| Dice | $\frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}} = \frac{2\,\mathrm{IoU}}{1 + \mathrm{IoU}}$ | 35, 40 |
| 칼만 필터(등속) | $\hat{x}_t = x_{t-1} + v_{t-1},\ x_t = \hat{x}_t + K(z_t - \hat{x}_t)$ | 42 |
| MOTA | $1 - \frac{\mathrm{FN} + \mathrm{FP} + \mathrm{IDSW}}{\mathrm{GT}}$ | 42 |
| IDF1 | $\frac{2\,\mathrm{IDTP}}{2\,\mathrm{IDTP} + \mathrm{IDFP} + \mathrm{IDFN}}$ | 42 |
| HOTA | $\frac{1}{19}\sum_{\alpha} \sqrt{\mathrm{DetA}_\alpha \cdot \mathrm{AssA}_\alpha}$ | 42 |

## 7부 학습 실무

| 이름 | 식 | 강 |
|---|---|---|
| YOLO 정규화 좌표 | $c_x = \frac{x_1 + x_2}{2W},\ w = \frac{x_2 - x_1}{W}$ ($c_y$, $h$는 $H$로) | 44 |
| 분광 섭동 증강 | $\rho_b' = g_b\,\rho_b + h_b + \epsilon$ | 45 |
| 믹스업 | $\tilde{x} = \lambda x_i + (1-\lambda)x_j,\ \tilde{y} = \lambda y_i + (1-\lambda)y_j$ | 45 |
| 코사인 스케줄 | $\eta_t = \frac{\eta_0}{2}(1 + \cos\frac{\pi t}{T})$ | 46 |
| 유효 배치 | GPU 하나의 배치 × 누적 횟수 × GPU 수 | 47 |
| 아핀 양자화 | $q = \operatorname{clip}(\operatorname{round}(x/s) + z, -128, 127),\ \hat{x} = s(q - z)$ | 50, 60 |
| 지오트랜스폼(화소 중심) | $X = x_0 + (c + 0.5)\Delta x,\ Y = y_0 + (r + 0.5)\Delta y$ | 50 |

## 8부 모델 진단

Diagnostics가 새로 정의한 식에는 이름 뒤에 "(프로젝트 정의)"를 붙였음. 표시가 없는 식은 논문·널리 쓰는 도구에서 온 업계 표준이거나 일반 수학임. 임계값은 식에 넣지 않았고, 범위와 잠정치 여부는 각 강에서 확인함.

| 이름 | 식 | 강 |
|---|---|---|
| 처방 규칙 판정 (프로젝트 정의) | $m \ge \tau$ 또는 $m < \tau,\ \tau \in [\tau_{\text{lo}}, \tau_{\text{hi}}]$ | 51 |
| 오차 유형별 AP 기여(TIDE 계열) | $\Delta \mathrm{AP}_k = \mathrm{AP}_{\text{oracle}(k)} - \mathrm{AP}_{\text{base}}$ | 52 |
| 위치 오차의 중심·크기 성분 (프로젝트 정의) | $\Delta_{\text{center}} = \frac{\lVert \mathbf{c} - \mathbf{c}^* \rVert}{\sqrt{w^* h^*}},\ \Delta_{\text{scale}} = \lvert \ln\frac{w}{w^*} \rvert + \lvert \ln\frac{h}{h^*} \rvert$ | 52 |
| 교차 클래스 억제율 CCSR (프로젝트 정의) | $\frac{\text{다른 클래스 상자에 지워진 NMS 전 TP 수}}{\text{NMS 전 TP 수}}$ | 53 |
| OFI (프로젝트 정의) | $1 - \frac{\operatorname{mean}_j(-\sum_k p_k \ln p_k)}{\ln K}$, $p_k$는 로짓 $s_k$의 소프트맥스 | 53 |
| 경계 오차 비율 BER (프로젝트 정의) | $\mathrm{BER}_c = \frac{\lvert E_c \cap B(Y_c) \rvert}{\lvert E_c \rvert},\ B(Y_c) = (Y_c \oplus D) \setminus (Y_c \ominus D)$ | 53 |
| 바서슈타인 거리(1차원) | $W_1 = \int \lvert F_{\text{학습}}(x) - F_{\text{대상}}(x) \rvert \, dx$ | 54 |
| 최대 평균 불일치 | $\mathrm{MMD}^2 = \mathbb{E}[k(a,a')] - 2\,\mathbb{E}[k(a,b)] + \mathbb{E}[k(b,b')]$ | 54 |
| ECE·MCE | $\sum_{m} \frac{n_m}{N}\lvert \mathrm{acc}_m - \mathrm{conf}_m \rvert$, $\max_m \lvert \mathrm{acc}_m - \mathrm{conf}_m \rvert$ | 55 |
| 온도 스케일링 | $p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$ | 55 |
| 불확실성 분해(MC 드롭아웃) | 우연적 $\frac{1}{T_s}\sum_t H(\mathbf{p}^{(t)})$, 인지적 $H(\bar{\mathbf{p}}) - \frac{1}{T_s}\sum_t H(\mathbf{p}^{(t)})$ | 55 |
| 유효 랭크 | $\mathrm{Rank}_{\mathrm{eff}} = \exp(-\sum_k p_k \ln p_k),\ p_k = \sigma_k / \sum_i \sigma_i$ | 56 |
| 특성 용량 활용률 CUR (프로젝트 정의) | $\mathrm{CUR} = \mathrm{Rank}_{\mathrm{eff}} / C$ | 56 |
| 비활성 채널 비율 DCR (프로젝트 정의) | $\frac{1}{C}\sum_{c=1}^{C} \mathbf{1}(\mu_c < \epsilon)$ | 56 |
| 수용영역-객체 스케일 불일치 RFM (프로젝트 정의) | $\ln\frac{2R_{95}}{S},\ S = \sqrt{w^2 + h^2}$ | 56 |
| 손상 오차 CE·mCE | $\mathrm{CE}_k = \frac{\sum_{s} E_{k,s}}{\sum_{s} E^{\text{base}}_{k,s}},\ \mathrm{mCE} = \frac{1}{10}\sum_k \mathrm{CE}_k$ | 57 |
| 강건성 감쇠 지수 RDI (프로젝트 정의) | $\mathrm{RDI}_k = \frac{A_{\text{clean}} - \frac{1}{5}\sum_{s} A_{k,s}}{A_{\text{clean}}}$ | 57, 61 |
| 낙차비·모달리티 편향 MBR (프로젝트 정의) | $\mathrm{DR}_m = \frac{v(N) - v(N \setminus \{m\})}{v(N)},\ \mathrm{MBR} = \frac{\mathrm{DR}_{\mathrm{EO}}}{\mathrm{DR}_{\mathrm{SAR}}}$ | 58 |
| 섀플리 값 | $\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{\lvert S\rvert!\,(\lvert N\rvert-\lvert S\rvert-1)!}{\lvert N\rvert!}[v(S\cup\{i\}) - v(S)]$ | 58 |
| 개입 분포 | $P(Y \mid \mathrm{do}(C=c))$ (관찰 $P(Y \mid C=c)$와 다름) | 59 |
| 평균 인과 맥락 효과 ACE (프로젝트 정의) | $\mathrm{ACE}_C = \frac{1}{N}\sum_{i}(\mathrm{Conf}_i(X_i) - \mathrm{Conf}_i(\tilde{X}_i))$ | 59 |
| 불변 위험 최소화(IRM) | $\min_{\Phi} \sum_{e} R^{e}(w \cdot \Phi) + \lambda \sum_{e} \lVert \nabla_{w \mid w=1.0} R^{e}(w \cdot \Phi) \rVert^2$ | 59 |
| 맥락 편향비 CBR (프로젝트 정의) | $\frac{\mathrm{mAP}_{\text{in}} - \mathrm{mAP}_{\text{out}}}{\mathrm{mAP}_{\text{in}}}$ | 59 |
| 맥락 유령 환각율 CHR (프로젝트 정의) | $\frac{1}{N}\sum_{i}\mathbb{I}[\max_{b}\mathrm{IoU}(b, B_i) \ge 0.5 \wedge s_b \ge \tau]$ | 59 |
| 신호 대 양자화 잡음비 | $\mathrm{SQNR}_l = 10 \log_{10} \frac{\mathbb{E}[X_l^2]}{\mathbb{E}[(X_l - \hat{X}_l)^2]}$ | 60 |
| SmoothQuant 채널 배율 | $c_j = \frac{\max \lvert X_j \rvert^{\alpha}}{\max \lvert W_j \rvert^{1-\alpha}}$ | 60 |
| 루프라인 모델 | $P = \min(P_{\text{peak}},\ I \cdot B),\ I^* = P_{\text{peak}}/B$ | 60 |
| 오퍼레이터 융합 결손도 FDR (프로젝트 정의) | $\frac{T_{\text{비융합}} - T_{\text{융합}}}{T_{\text{융합}}}$ | 60 |
| 순 개선·회귀율 RR (프로젝트 정의) | $\Delta N_{\text{net}} = N_{01} - N_{10},\ \mathrm{RR} = \frac{N_{10}}{N_{11} + N_{10}} \times 100\%$ | 61 |
| 종합 배포 판정 (프로젝트 정의) | $\mathbb{I}(G_{\text{macro}} \land G_{\text{slice}} \land G_{\text{robust}} \land G_{\text{calib}} \land G_{\text{latency}})$ | 61 |

8부 식을 읽을 때 세 가지를 기억함. RDI는 하락률이라 **낮을수록 강건**하고, 61강 배포 게이트는 "max RDI ≤ 0.40", 곧 유지율 $1 - \mathrm{RDI} \ge 0.60$으로 씀. OFI의 $s_k$는 NMS 전 헤드 로짓이고, 0~1 신뢰도를 넣으면 값의 상한이 낮게 묶임(53강). 소형 표적은 길이가 아니라 **넓이(화소²)**로 판정함. 실제 넓이를 $\mathrm{GSD}^2$로 나눈 화소 넓이(EPS)가 16 미만(약 4×4화소)이면 형상 인식 한계, $32^2$ = 1,024 미만이면 COCO 소형이고, 대각선으로 재는 RFM과는 따로 봄(56·57강).
