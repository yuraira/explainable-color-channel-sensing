# 연구 이전 및 재현 안내

이 문서는 다른 컴퓨터에서 IFMBE 초록과 포스터 작업을 이어가기 위한 파일 구성과 재현 절차를 정리한 안내서임.

## 1. GitHub에 보존된 핵심 자료

- `glucose_ketone_dataset/`: 포도당 11장과 케톤 8장으로 구성된 원본 이미지 19장
- `src/`: 패치 검출, 색상 특징 추출, 머신러닝 학습, 설명가능성 분석 코드
- `scripts/`: 각 분석 단계의 결과 검증 코드
- `outputs/`: 발표용 그림, 결과 CSV, 실행 설정 JSON, 검증 보고서
- `requirements.txt`: Python 패키지와 버전
- `README.md`: 전체 연구 과정, 결과표, 해석 및 한계
- `IFMBE_abstract_plain.txt`: 293단어 영문 제출용 초록
- `IFMBE_abstract_draft.md`: 영문 초록과 한국어 번역본
- `IFMBE_이유라_초록.txt`: 영문과 한국어 초록 통합본

## 2. 의도적으로 GitHub에서 제외한 재생성 가능 파일

- `outputs/patch_detection/crops/`: 원본 이미지에서 다시 만들 수 있는 패치 이미지 1,824장
- `*.pt`, `*.pth`: 딥러닝 체크포인트
- `*.joblib`: 다시 학습할 수 있는 머신러닝 모델 바이너리
- `*.xlsx`: CSV와 중복되는 검토용 파일
- `.tmp/`: PPT 확인 등에 사용한 임시 파일

위 파일은 포스터 작성과 결과 확인에 필수적인 원자료가 아님. 원본 이미지, 코드, 분할표, 최종 수치, 설정 및 검증 보고서는 GitHub에 보존함.

## 3. 새 컴퓨터에서 저장소 받기

Windows PowerShell에서 다음 명령을 실행함.

```powershell
git clone https://github.com/yuraira/explainable-color-channel-sensing.git
Set-Location explainable-color-channel-sensing
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

본 연구를 수행한 Python 버전은 `3.12.0`임. 학회 포스터 작성만 수행하는 경우 저장된 그림과 CSV를 바로 사용할 수 있으므로 모델을 다시 학습할 필요가 없음.

## 4. 머신러닝 분석 재실행 순서

저장된 결과를 유지하려면 기존 `outputs` 폴더를 먼저 다른 이름으로 보관한 뒤 실행함.

```powershell
Rename-Item -LiteralPath outputs -NewName outputs_published

python src/detect_patches.py --input-dir glucose_ketone_dataset --output-dir outputs/patch_detection
python scripts/verify_patch_detection.py --output-dir outputs/patch_detection

python src/extract_color_features.py
python scripts/verify_color_features.py

python src/validate_color_features.py
python scripts/verify_feature_validation.py

python src/create_data_splits.py
python scripts/verify_data_splits.py

python src/train_ml_models.py
python scripts/verify_ml_models.py

python src/refit_selected_ml_models.py
python scripts/verify_selected_ml_models.py

python src/compare_reduced_feature_models.py
python scripts/verify_reduced_feature_models.py

python src/analyze_ml_explainability.py

python src/analyze_extra_trees_feature_efficiency.py
python src/analyze_random_forest_feature_efficiency.py
python scripts/verify_tree_feature_efficiency.py
```

각 생성 스크립트는 기존 출력 폴더를 덮어쓰지 않도록 설계되어 있음. 중간에 다시 실행해야 하면 해당 단계의 출력 폴더를 별도로 보관하거나 스크립트가 지원하는 `--overwrite` 옵션을 사용함.

## 5. 포스터 작성에 바로 사용할 자료

- 연구 제목과 초록: `IFMBE_abstract_draft.md`
- 전체 연구 방법과 결과표: `README.md`
- 패치 검출 결과: `outputs/patch_detection/qc_contact_sheet.jpg`
- RGB 및 HSV 변화: `outputs/feature_validation/figures/concentration_trends_rgb.png`, `concentration_trends_hsv.png`
- 교차검증 설계: `outputs/data_splits/figures/cross_validation_split_design.png`
- 모델 성능: `outputs/modeling/comparison/figures/model_performance_comparison.png`
- 예측값과 실측값: `outputs/modeling/comparison/figures/observed_vs_predicted.png`
- SHAP 특징 중요도: `outputs/explainability/ml/figures/color_channel_importance.png`
- SHAP 기여 방향: `outputs/explainability/ml/figures/shap_direction_summary.png`
- 단일 특징 비교: `outputs/modeling/reduced_features/figures/random_forest_single_vs_full_color_features.png`
- 트리 모델 경량화: `outputs/modeling/feature_efficiency/figures/extra_trees_feature_reduction_efficiency.png`, `random_forest_feature_reduction_efficiency.png`

## 6. 포스터에서 사용할 핵심 수치

| 분석물 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 포도당 | 0.351 mg/mL | 0.912 mg/mL | 0.981 |
| 케톤 | 0.041 mg/mL | 0.284 mg/mL | 0.991 |

- RGB 모델의 주요 특징: 두 분석물 모두 G
- HSV 모델의 주요 특징: 두 분석물 모두 Hue
- 포도당 설명 방향: 높은 G가 높은 농도 예측에 기여
- 케톤 설명 방향: 낮은 G와 Hue가 높은 농도 예측에 기여
- 검증 범위: 동일 원본 이미지 안의 96개 웰 위치를 그룹화한 내부 교차검증
- 주요 한계: 각 농도당 원본 이미지가 1장이므로 새로운 플레이트, 촬영 조건 또는 환자 검체에 대한 외부 일반화는 확인하지 않음

## 7. 컴퓨터 이전 전 확인 사항

- GitHub 저장소에서 원본 이미지 19장이 모두 열리는지 확인
- GitHub README에서 모든 그림이 표시되는지 확인
- 새 컴퓨터에서 `requirements.txt` 설치 가능 여부 확인
- 학회 포스터 원본 파일은 완성 후 GitHub 또는 별도 클라우드에 추가 보관
- 환자정보나 개인정보가 포함된 자료는 공개 저장소에 업로드하지 않음
