# Skin cancer detection using 3D-TBP

3D-TBP를 이용한 피부암 탐지


## Dependencies
- [Polars](https://pola.rs/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [timm](https://timm.fast.ai/)
- [Pytorch](https://pytorch.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Albumentation](https://albumentations.ai/docs/)
- [h5py](https://docs.h5py.org/en/stable/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)


## Dataset

데이터셋은 [Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge)에서 수집했습니다.

데이터가 대용량이므로, 개인적으로 Kaggle API 키를 발급받고 .zip 확장자를 불러오고 구글 드라이브에 압축을 풀고 사용하세요.

훈련 데이터는 이미지 데이터와 메타 데이터로 구성되어있습니다.

* #### 이미지 데이터 : '양성' 이미지 400666장 및 '악성' 이미지 393장
<img width="400" alt="image" src="https://github.com/user-attachments/assets/7bf0e0db-e80b-42d3-b862-b05e236525ff">

* #### 메타 데이터 : 501059개 레코드 및 55개 필드로 구성된 데이터프레임


## 평가 지표 (Evaluation Metrics)

* #### pAUC (ROC곡선 아래의 부분 면적)
  <img width="300" alt="image" src="https://github.com/user-attachments/assets/7765dd15-1024-4e03-b5d8-190cee71a1a4">


## 피부암 탐지 시스템
* ### 학습 데이터 구축
  
  * #### 이미지 전처리
    * ##### 머리카락(털, hair) 제거
    * ##### 배경 제거
    * ##### 마커/잉크 제거

    * ##### Before (처리 전) :
      <img width="500" alt="image" src="https://github.com/user-attachments/assets/a606d5aa-b9a4-44c7-8ae2-76ad42a0ec46">
        
    * ##### After (처리 후) : 
      <img width="500" alt="image" src="https://github.com/user-attachments/assets/e931e572-e5e2-4d56-99f7-b175b772671b">
    
  * #### 메타 데이터 처리
    * ##### 피처 엔지니어링 -> 새로운 분석 지표 44개를 추가적으로 생성
    * ##### 결측치 처리
    * ##### 카테고리 변수 인코딩
    * ##### 수치형 변수 스케일링
    * ##### 적대적 검증 (Adversarial Validation)
      
  * #### 이미지 데이터 증강
    * ##### 전환(Transpose)
    * ##### 밝기 및 대비 조정(RandomBrightnessContrast)
    * ##### 모자이크 처리(CoarseDropout)
    * ##### 블러 처리(Motion Blur, Median Blur, Gaussian Blur)
    * ##### 왜곡(Optical Distortion, Grid Distortion, Elastic Transform)
    * ##### ShiftScaleRatate
    * ##### 대비 제한 적응형 히스토그램 평활화(CLAHE)
    * ##### 색상 조정(Hue, Saturation, Value)
    * ##### Microscope
    * ##### 정규화(Normalization)
  
  * #### Result (결과) :
      <img width="500" alt="image" src="https://github.com/user-attachments/assets/36f67863-107a-41b1-b736-331debb74a2b">


* ### 멀티모달(Multi-model) 신경망 네트워크
  * #### CNN(EfficientNet-B3) 및 NN
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/f44dc39d-8ef7-4fe3-9bf6-a5041d5acd20">


## 학습 결과

에포크(Epoch) 1만 돌렸을 때 pAUC 점수는 약 0.167[0.0, 0.2]

## 연구 논문

[Skin Cancer Classification using Deep Learning](https://drive.google.com/file/d/1hsKeB9cscI6wswlgyTnQX5Il1fZC6LpP/view)

## License
[MIT LICENSE](https://github.com/paulms77/Skin-cancer-detection-using-3D-TBP/blob/main/LICENSE)
