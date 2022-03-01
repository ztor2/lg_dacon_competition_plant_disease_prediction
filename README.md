## 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회
### 최종 순위 3위 & Private score 6위(0.95324) 제출 코드 및 설명
- 주최 및 주관: LG AI Research, 데이콘
- [대회 페이지](https://dacon.io/competitions/official/235870/overview/description)

#### Edit log
- torch_aug_with_part_bbx 함수 indentation 오류 수정, 불필요 변수(labels_str_oversampled) 제거(22.02.06)
- 참고문헌 추가(22.02.24)
- 최종 제출 코드 업로드, 최고 submission score를 달성한 학습 완료된 파라미터 파일 업로드(22.03.01)
<br>

#### 개발 환경

- 모든 실험은 Google Colab Pro Plus 환경에서 수행되었습니다.\
OS: Linux-5.4.144+-x86_64-with-Ubuntu-18.04-bionic\
python==3.7.12\
json==2.0.9\
opencv-python==4.1.2.30\
pandas==1.3.5\
numpy==1.19.5\
matplotlib==3.2.2\
tqdm==4.62.3\
PIL==7.1.2\
scipy==1.4.1\
torch==1.10.0+cu111\
torchvision==0.11.1+cu111\
skimage==0.18.3\
sklearn==1.0.2\
timm==0.5.4
- 전처리 포함 학습 완료 시간은 주어진 개발 환경에서 80 epoch 기준 약 130분입니다.
- 해당 코드는 코드가 저장된 경로에 data 폴더가 있고, 그 안에 train, test 폴더 형태로 데이터가 주어져있다고 가정합니다.
- 또한, '리더보드 점수 복원이 가능한 학습이 완료된 모델'과 submission template 파일인 sample_submission.csv 파일이 data 폴더 내에 포함되어 있다고 가정합니다.
- 학습이 완료된 모델과 추론이 완료된 submission 파일은 data 폴더 내에 저장됩니다.
- 코드 실행 전, root 경로를 코드가 저장된 경로로 변경하여야 합니다.

**train.ipynb 실행 방법**
1. 첫 셀에서 모델 저장 경로, 모델 저장 이름 설정.
2. 이후의 모든 셀 실행.

**test.ipynb 실행 방법**
1. 첫 셀에서 저장된 모델 경로, 저장된 모델 이름, 저장할 submission 파일 이름 설정.
2. 이후의 모든 셀 실행.
<br>

#### 환경 데이터

- 환경데이터의 경우, 서로 다른 종 식물들의 환경 데이터가 섞여 있는 점을 고려하여 원 데이터를 변화량 데이터로 바꾸어(dataframe.diff() 기능 이용) 학습에 이용했습니다. 
- 각 측정치의 최고, 최저, 평균 정보가 크게 다른 점이 없다고 판단해 평균 데이터만을 학습에 이용했습니다. 결측치가 많은 3개 측정치 이외의 데이터는 사용하지 않았습니다. 이상치의 영향을 최소화하기 위해 robust scaler를 사용했습니다.
- 환경 데이터에 대한 증강은 각 데이터 값에 무작위 노이즈 값을 곱해주는 scaling 함수만을 사용했습니다. 
<br>

#### 이미지 증강 함수

- 사용된 이미지 증강 함수는 코드에서 확인하실 수 있으며, 기본적인 증강 함수 외에 추가한 부분은 edge detection을 이용한 증강 함수와 bounding box specific 증강 함수, mixup(Zhang et al., 2017) 정도입니다.
- edge 증강의 경우 openCV의 Canny, Sobel, Laplacian 함수 중 하나로 edge를 detection한 후 원래 이미지와 합치는 방식으로 증강을 수행합니다.
- bounding box specific 증강 함수는 이미지를 카피하여 랜덤하게 이미지 증강을 수행하고, 원래 이미지에 증강한 이미지의 bounding box 부분만(또는 bounding box 바깥 부분만) 잘라 붙이는 방식으로 증강을 수행합니다. 기본 bounding box 외에도 병해 부위에 대한 bounding box에도 같은 방식으로 증강을 수행했습니다. 이를 편의상 target augmentation이라 명명했습니다.
- Cutmix와 random erase 함수도 사용했었는데, 테스트 결과 validation 성능은 올라가지만 submission 시 성능이 하락하는 현상이 관찰되어 배제했습니다. 
- 전반적으로 증강에 있어 무작위성을 최대한 많이 부여하려 했습니다. 그리고 데이터 imbalance를 고려해 일정 비율로 데이터 oversampling을 수행했습니다.
- [이미지 증강 example](https://dacon.io/competitions/official/235870/talkboard/405943?page=1&dtype=recent)
<br>

#### 학습 전략 및 파라미터 설정

- 학습 모델은 timm 라이브러리의 efficientnet 계열 모델 중 하나인 pre-trained tinynet_a(Han et al., 2020) 모델을 사용했고, 시계열 데이터에는 baseline 코드의 LSTM 모델을 그대로 사용했습니다. Tinynet이 특별히 성능이 더 우수하기보다는 다른 모델에 비해 loss 변화가 안정적으로 관찰되어 사용했습니다. 

- 일정 수준의 score를 달성한 이후 validation score와 submission score가 비례하지 않는 현상이 있어 불량 데이터가 상당수 있다고 생각했습니다. 다른 참가자분들도 EDA를 통해 이에 대한 문제를 제기한 것으로 알고 있습니다. 저의 경우 label smoothing 수치를 극단적으로 높게 주어 모델의 confidence를 최대한 낮추는 방향으로 접근했습니다. Label 수를 고려했을 때에 0.9-0.95까지 label smoothing 값을 줄 수 있다고 판단해 설정했고, validation score가 98을 넘지 않도록 epoch이나 learning rate를 설정했습니다. 이로써 train 데이터에 대한 과적합을 최대한 피하고자 했고, 97 중후반대의 validation score에서 가장 높은 submission score를 확인할 수 있었습니다.
<br>

#### 모델 한계

- 무작위성이 강한 이미지 증강 함수에 의존하는 코드 구조이다 보니 성능의 편차가 큰 편입니다. 예상치 못하게 높은 성능이 나올 수 있다는 것이 competition에서는 장점이 된다고도 할 수 있을 것 같습니다. 다만, best score가 잘 복원되지 않는 부분은 아쉬웠습니다. Public score 기준 평균적으로 94.8 – 95.1 사이의 성능이 가장 많이 관찰되었습니다.
- 또한, 시간적인 한계로 다양한 ensemble 전략이나 모델에 대한 실험을 수행하지는 못했습니다. 
- 불량 데이터에 대한 좀 더 근본적인 해결 전략이 있을 것 같은데 그 점을 깊게 조사해보지 못한 것도 아쉬운 부분입니다. 다른 참가팀에서 더 좋은 해결 전략을 제시해 주실 것 같습니다.
<br>

#### 참고문헌
- Han, K., Wang, Y., Zhang, Q., Zhang, W., Xu, C., & Zhang, T. (2020). Model rubik’s cube: Twisting resolution, depth and width for tinynets. Advances in Neural Information Processing Systems, 33, 19353-19364.
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.
- https://github.com/uchidalab/time_series_augmentation/blob/master/example.ipynb

