## 시각-언어 모델을 활용하는 제로샷 낙상 감지 방법론 (Zero-Shot Fall Detection: Leveraging Text Prompt in Vision-Language Models)

Koh Young Technology와 협업한 산학 프로젝트

<br>

##  Members
<div align="center">



김준용 | 길다영 
:-:|:-:
<a href="https://github.com/wragon"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"></a>[![junyong@soongsil.ac.kr](https://img.shields.io/badge/Mail-004788?style=for-the-badge&logo=maildotcom&logoColor=white&link=mailto:junyong@soongsil.ac.kr)](junyong@soongsil.ac.kr)|<a href="https://github.com/arittung"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"></a>[![dayoung.k.ssu@gmail.com](https://img.shields.io/badge/Mail-004788?style=for-the-badge&logo=maildotcom&logoColor=white&link=mailto:dayoung.k.ssu@gmail.com)](mailto:dayoung.k.ssu@gmail.com)
<img alt="Html" src ="https://img.shields.io/badge/팀장-B1BED5?style=for-the-badge"/>|<img alt="Html" src ="https://img.shields.io/badge/팀원-B1BED5?style=for-the-badge"/>


</div>

<br>



<p align="center">
  
![ezgif com-resize](https://github.com/VIP-Projects/Zero-Shot-Fall-Detection/assets/53934639/3fd9d485-217a-428f-9c21-7de9ea47538f)

</p>

## 프로젝트 소개

### 📅 개발 기간
#### [2023 산학 프로젝트 챌린지] : 2023.08.30(Wed) - 2023.11.21(Tue)
<br>

### 💡 개발 배경 및 목적

#### Koh Young Technology

- 제조 자동화 산업 
- <b>이상 감지 검출</b> 연구 및 개발의 한계
  - 제조 및 검증 과정에서 고장 부품을 학습하고 선별하는 것이 어려움.
  - 모든 고장 부품에 대한 충분한 양의 훈련 데이터셋 얻기 어려움.
  - Unseen 영상 및 고장 카테고리 변경 시 재학습이 필요함.

- 고장 카테고리를 학습하여 정확도를 1-2% 향상시키는 것보다 <b>Zero-Shot 방법론을 통해 학습 비용 감축</b>에 관심을 보임.

#### 목적

- <b>시각-언어 모델을 이용</b>하여 <b>데이터셋을 훈련없이 Abnormal Detection이라는 일반적인 프레임워크</b> 만드는 것이 목표.

- 기존 연구인 Fall Detection을 사례연구로 먼저 시도했고, <b>다양한 상황에서의 이상 감지로 일반화 가능성</b>을 확인함. 

<br><br>

## 시스템 구성

<p align="center">
 <img src="https://github.com/VIP-Projects/Zero-Shot-Fall-Detection/assets/53934639/eac17a36-caa5-465d-8c00-57fdbdc94b68" width="500px"> </p>

<center>

Rule-based|Zero-Shot
:--:|:--:
<img src="https://github.com/VIP-Projects/Zero-Shot-Fall-Detection/assets/53934639/c367c6c7-41e0-48c6-a50f-7485880258c0" width="500px">|<img src="https://github.com/VIP-Projects/Zero-Shot-Fall-Detection/assets/53934639/2a7614fc-97bd-4506-83d9-4bd130ce94ae" width="500px">
OpenPose<br>가속도|BLIP<br>GroundingDINO<br>Human-Object Interaction(HOI)

</center>

<br><br>
### 💫 기대 효과

- 빠른 신고와 조치가 필요한 다양한 상황에서 정교화된 Prompt 기술로 <b>학습 없이 즉각적인 대응이 가능한 시스템 구축</b>
- 노인 및 기타 취약한 인구에게 생활의 편의성을 제공. 스스로 살아가는 데 도움을 주고, 가족이나 감독자에게 심적 안정감을 제공
- 낙상 예방과 안전 강화: 낙상을 빠르게 감지하고 경고하는 시스템을 통해 부상 예방 및 심각성을 감소시키는 데 도움
- 의료 모니터링 향상: 의료 모니터링 시스템과 통합하여, 환자의 건강 추적 및 치료 계획에 도움을 제공
- 스마트 홈과 IoT 통합: 스마트 홈 및 IoT 기기와 통합하여 스마트 홈 시스템의 지능 고도화를 통해, 홈 자동화 및 편의성 증가
- 스마트 팩토리와 센서 기반 안전 시스템과 통합: 제로샷 낙상 감지 기술은 작업자가 기계나 로봇과 함께 작업시 낙상 사고를 감지하고 경고. 센서 기기와 통합하여 더 <b>효과적인 안전 시스템을 구축</b>하는 데 활용
- 데이터 수집 및 학습 비용 감소: <b>수집하고 학습하지 못한 문제 상황에도 Zero-shot 기반의 일반적인 방법론을 통해 데이터 수집 및 학습 비용 절감</b>이 가능
- 스마트 팩토리 자동화 및 최적화: Zero-shot 감지를 통해 <b>학습되지 않은 불량을 검출하여 제조 자동화 및 최적화에 기여</b>





<br><br>


### 🛠 개발 환경 및 개발 언어

```
운영체제 : Linux 20.04
GPU : GeForce RTX 3090
개발언어 : Python 3.8
사용 툴 : Linux
AI 라이브러리 : Pytorch 1.8.1
```


<br>

### 🗂 시스템 디렉터리 구조

<details>
<summary>Directory Structure</summary>
<div markdown="1">

  ```
virtual fitting dir

.
|--Android App
|      |--annotation
|      |      '--image_caption.json
|      |--BLIP
|      |      |--config
|      |      |      '--med_config.json
|      |      |--models
|      |      |      |--__init__.py
|      |      |      |--blip.py
|      |      |      |--med.py
|      |      |            '--vit.py
|      |      |--weights
|      |      |      '--model_base_capfilt_large.pth
|      |--GroundingDINO
|      |      |--groundingdino
|      |      |      |--config
|      |      |      |      |--__init__.py
|      |      |      |      '--GroundingDINO_SwinT_OGC.py
|      |      |      |--datasets
|      |      |      |      |--__init__.py
|      |      |      |      '--transforms.py
|      |      |      |--models
|      |      |      |      |--GroundingDINO
|      |      |      |      |      |--backbone
|      |      |      |      |      |      |--__init__.py
|      |      |      |      |      |      |--backbone.py
|      |      |      |      |      |      |--position_encoding.py
|      |      |      |      |      |      '--swin_transformer.py
|      |      |      |      |      |--scrc
|      |      |      |      |      |      |--cuda_version.cu
|      |      |      |      |      |      '--vision.cpp
|      |      |      |      |      |--__init__.py
|      |      |      |      |      |--bertwarper.py
|      |      |      |      |      |--fuse_modules.py
|      |      |      |      |      |--groundingdino.py
|      |      |      |      |      |--ms_deform_attn.py
|      |      |      |      |      |--transformer.py
|      |      |      |      |      |--transformer_vanilla.py
|      |      |      |      |      '--utils.py
|      |      |      |      |--__init__.py
|      |      |      |      '--registry.py
|      |      |      |--util
|      |      |      |      |--__init__.py
|      |      |      |      |--box_ops.py
|      |      |      |      |--get_tokenlizer.py
|      |      |      |      |--inference.py
|      |      |      |      |--logger.py
|      |      |      |      |--misc.py
|      |      |      |      |--slconfig.py
|      |      |      |      |--slio.py
|      |      |      |      |--time_counter.py
|      |      |      |      |--utils.py
|      |      |      |      |--visualizer.py
|      |      |      |      '--vl_utils.py
|      |      |      '--__init__.py
|      |      |--weights
|      |      |      '--groundingdino_swint_ogc.pth
|      |--abnormal_detection.py
|      |--Image_caption.py
|      |--json_sort.py
|      |--makeVideo.py
|      '--video2img.py
'
  ```

</div>

</details>

<br>


## 실행 방법

```
# 1. 환경 설정(Package 설치)
pip install -r requirements.txt

# 2. BLIP 모델과 GroundingDINO 모델 Download
pretrained model Download

# 3. 이미지 캡셔닝 실행
python Image_caption.py

# 4. json file 정렬
python json_sort.py

# 5. 이상 감지 실행
python abnormal_detection.py
```



<br><br>
