# IFMBE Abstract Draft

## Title

**Explainable Color-Channel Modeling for Glucose and Ketone Sensing**

**포도당 및 케톤 센싱을 위한 설명 가능한 색상 채널 모델링**

## English Abstract

Smartphone colorimetric sensing is convenient, but models may exploit illumination or background variation rather than sensor color changes. We evaluated whether interpretable color-channel models could predict glucose and ketone concentrations with reproducible feature-level explanations. Nineteen source images spanning 11 glucose and 8 ketone concentrations were divided into 1,824 circular patches. Median RGB and circular hue, saturation, and value features were extracted from central regions after excluding specular highlights. Five regressors were compared with a lightweight convolutional neural network using nested five-fold cross-validation grouped by 96 well positions. Model selection was confined to inner folds, and performance was measured on held-out positions. Nested-selected color models achieved mean absolute errors of 0.351 mg/mL for glucose and 0.041 mg/mL for ketone, compared with 0.408–0.468 and 0.144–0.147 mg/mL for the image models. Paired well-position bootstrap intervals favored the color models. TreeSHAP and permutation importance identified green and hue as the leading features within RGB-only and HSV-only models for both analytes in all five folds. Higher green values contributed to higher glucose predictions, whereas lower green and hue values contributed to higher ketone predictions, matching the observed concentration trends. Full-patch CNNs concentrated 44.6% and 40.4% of absolute Grad-CAM sensitivity within central regions occupying approximately 27% of each patch while retaining substantial sensitivity outside them. A background-only model retained concentration-associated information for ketone, indicating possible acquisition confounding. Because only one source image was available per concentration, these findings represent internal position-grouped validation and do not establish generalization to new plates, imaging conditions, or unmeasured concentrations. Color-channel modeling provided interpretable predictions and model-reliance evidence, emphasizing the need for independent imaging repeats.

**Keywords:** colorimetric sensor; explainable modeling; glucose; ketone; SHAP; color channels

## 국문 검토본

비색센서는 스마트폰 영상으로 간편하게 농도를 측정할 수 있지만, 예측 모델이 화학적으로 관련된 색 변화 대신 조명이나 배경 변화에 의존할 가능성이 있음. 본 연구는 설명 가능한 색상 채널 모델이 포도당과 케톤 농도를 예측하면서 특징 수준의 재현 가능한 근거를 제시할 수 있는지 평가함. 포도당 11개 농도와 케톤 8개 농도를 나타내는 원본 이미지 19장에서 원형 센서 패치 1,824개를 추출함. 반사광을 제외한 센서 중앙 영역에서 RGB 중앙값과 원형 표현 Hue, Saturation, Value 특징을 계산함. Ridge, Elastic Net, SVR, Random Forest, Extra Trees를 경량 CNN과 비교했으며, 96개 센서 위치를 그룹으로 묶은 nested 5-fold 교차검증을 적용함. 내부 fold에서만 모델을 선택하고 분리된 위치 그룹에서 성능을 평가한 결과, 색상 특징 모델의 MAE는 포도당 0.351 mg/mL와 케톤 0.041 mg/mL였으며 이미지 모델은 각각 0.408–0.468 mg/mL와 0.144–0.147 mg/mL였음. 위치를 짝지은 bootstrap에서도 색상 특징 모델의 낮은 MAE가 일관되게 나타남. TreeSHAP과 permutation importance는 두 분석물 모두에서 G와 Hue를 5개 fold의 주요 특징으로 동일하게 제시함. 높은 G는 높은 포도당 예측에 기여했으며 낮은 G와 Hue는 높은 케톤 예측에 기여하여 관찰된 농도별 색 변화 방향과 일치함. 전체 패치 CNN은 약 27% 면적의 중앙 영역에 포도당 44.6%와 케톤 40.4%의 절댓값 Grad-CAM 민감도를 집중했지만 중앙 바깥에도 상당한 민감도가 남음. 케톤의 배경-only 모델에도 농도 관련 정보가 남아 촬영 교란 가능성이 확인됨. 농도별 원본 이미지가 1장뿐이므로 본 결과는 위치 그룹 내부 검증이며 새로운 플레이트, 촬영 조건 또는 미측정 농도에 대한 일반화를 입증하지 않음. 본 결과는 색상 채널 모델링이 비색센서 예측의 판단 근거를 진단하는 정확하고 설명 가능한 프레임워크가 될 가능성을 보여주며 독립 반복 촬영의 필요성을 강조함.

## 제출 전 확인 사항

- 실제 학회 초록의 단어 수·구조·키워드 수에 따른 형식 조정 필요
- 저자명·소속·교신저자 정보 추가 필요
- 독립 이미지 검증이 아닌 위치 그룹 내부 검증이라는 표현 유지 필요
- “화학 반응 입증” 대신 “관찰된 색 변화와 기여 방향의 일치”라는 결론 유지 필요
