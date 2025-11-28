# 퀀트 전략 모델 개발 프로젝트

S&P 500 시장 데이터를 기반으로 **변동성 조절 샤프 지수(Volatility-Adjusted Sharpe Ratio)**를 최대화하는 안정적인 투자 전략 모델 개발

## 🎯 프로젝트 목표

**"대박 수익률이 아니라 시장보다 조금 더 잘하면서 훨씬 안정적인 모델"**

- **핵심 철학**: Low Risk, Medium Return (안정성 우선)
- **사용 모델**: LightGBM (정형 데이터에 강함)
- **평가 지표**: Adjusted Sharpe Ratio with Volatility & Return Penalty

## 📊 데이터 구조

### Train Data

- **Shape**: 9,021 rows × 98 columns
- **Features**: M*, E*, I*, P*, V*, S*, MOM*, D* (8개 카테고리)
- **Target**:
  - `forward_returns`: 일일 S&P 500 수익률
  - `risk_free_rate`: 연방기금 금리
  - `market_forward_excess_returns`: 시장 초과 수익률

### Test Data

- **Shape**: 10 rows × 99 columns
- **Special Columns**:
  - `is_scored`: 평가 대상 여부
  - `lagged_forward_returns`: 1일 지연된 수익률
  - `lagged_risk_free_rate`: 1일 지연된 무위험 금리
  - `lagged_market_forward_excess_returns`: 1일 지연된 시장 초과 수익률

## 🛠️ 개발 단계

### Phase 0: 데이터 분석 ✅

- [x] 데이터 로드 및 기본 정보 확인
- [x] EDA notebook 생성
- [ ] 데이터 분석 완료

### Phase 1: 베이스라인 모델

- [ ] 평가 함수 구현
- [ ] Position = 1.0 (시장 추종) 전략 구현
- [ ] 베이스라인 점수 확인

### Phase 2: 알파 예측 모델

- [ ] 데이터 전처리 파이프라인
- [ ] 피처 엔지니어링
- [ ] LightGBM 학습 (Target: Alpha = return - market_return)
- [ ] 시계열 교차 검증

### Phase 3: 리스크 관리

- [ ] 변동성 기반 후처리 로직
- [ ] 포지션 크기 동적 조절
- [ ] 최종 제출 파일 생성

## 📂 프로젝트 구조

```
quant_strategy/
├── data/
│   ├── raw/              # 원본 데이터 (train.csv, test.csv)
│   └── processed/        # 전처리된 데이터
├── notebooks/            # Jupyter notebooks (EDA, 실험)
│   └── 01_eda.ipynb
├── src/                  # 소스 코드
│   ├── metric.py         # 평가 함수
│   ├── preprocessing.py  # 데이터 전처리
│   ├── features.py       # 피처 엔지니어링
│   └── models.py         # 모델 학습/예측
├── scripts/              # 실행 스크립트
│   ├── baseline.py       # Phase 1: 베이스라인
│   ├── train.py          # Phase 2: 모델 학습
│   └── predict.py        # Phase 3: 예측 및 제출
├── results/              # 분석 결과, 로그
├── models/               # 학습된 모델 저장
└── conf/                 # 설정 파일
    └── config.yaml
```

## 🎓 평가 지표 (Competition Metric)

```python
# Adjusted Sharpe Ratio
adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

# Sharpe Ratio
sharpe = strategy_mean_excess_return / strategy_std * sqrt(252)

# Volatility Penalty (최대 1.2배까지 허용)
vol_penalty = 1 + max(0, (strategy_vol / market_vol) - 1.2)

# Return Penalty (시장 대비 낮은 수익률 페널티)
return_gap = max(0, (market_return - strategy_return) * 100 * 252)
return_penalty = 1 + (return_gap^2) / 100
```

### 제약 조건

- **Position 범위**: 0 ~ 2 (최대 2배 레버리지)
- **Penalty**: 변동성 > 시장 × 1.2 또는 수익률 < 시장 수익률 시 점수 감소

## 🚀 시작하기

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 2. EDA 실행

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. 베이스라인 실행 (Phase 1)

```bash
python scripts/baseline.py
```

## 📈 개발 전략

1. **보수적 접근**: 단순한 전략부터 시작 (Position = 1.0)
2. **점진적 개선**: 알파 예측 모델 추가
3. **리스크 관리**: 변동성 높을 때 포지션 축소
4. **검증 철저**: Time Series CV로 미래 데이터 누출 방지

## 📝 참고 사항

- `is_scored=False`인 데이터는 학습/평가에서 제외
- 결측치가 많은 초기 데이터 처리 전략 필요
- 시계열 데이터이므로 데이터 누출 주의

---

**Last Updated**: 2025-11-27
