## 시각-언어 모델을 활용하는 제로샷 낙상 감지 방법론 (Zero-Shot Fall Detection: Leveraging Text Prompt in Vision-Language Models)

#### [2023 산학 프로젝트 챌린지] : 2023.08.30(Wed) - 2023.11.21(Tue)

<br>

##  Members
<div align="center">



김준용 | 길다영 
:-:|:-:
<a href="https://github.com/wragon"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"></a>[![junyong@soongsil.ac.kr](https://img.shields.io/badge/Mail-004788?style=for-the-badge&logo=maildotcom&logoColor=white&link=mailto:junyong@soongsil.ac.kr)](junyong@soongsil.ac.kr)|<a href="https://github.com/arittung"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"></a>[![dayoung.k.ssu@gmail.com](https://img.shields.io/badge/Mail-004788?style=for-the-badge&logo=maildotcom&logoColor=white&link=mailto:dayoung.k.ssu@gmail.com)](mailto:dayoung.k.ssu@gmail.com)
<img alt="Html" src ="https://img.shields.io/badge/팀장-B1BED5?style=for-the-badge"/>|<img alt="Html" src ="https://img.shields.io/badge/팀원-B1BED5?style=for-the-badge"/>


</div>

<br><br>

## 프로젝트 소개
Koh Young Technology와 협업한 산학 프로젝트

<p align="center">
  
![ezgif com-resize](https://github.com/VIP-Projects/Zero-Shot-Fall-Detection/assets/53934639/3fd9d485-217a-428f-9c21-7de9ea47538f)

</p>

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
OpenPose<br>가속도|Blip<br>GroundingDINO<br>Human-Object Interaction(HOI)

</center>



