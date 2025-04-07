# MK's Camera Calibration

## 📌 개요
이 프로그램은 OpenCV를 사용하여 촬영한 영상을 Camera Calibration하는 프로그램입니다.   

## :memo: 기능 소개
- ✅ **Camera calibration**
```python
 # 이미지 로드
 img = cv2.imread('img.png')

 # 이미지를 회색 스케일로 변환
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 # 노이즈를 줄이기 위해 중앙값 블러 적용
 gray = cv2.medianBlur(gray, 5)

 # 적응형 임계값을 사용하여 가장자리 감지
 edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

 # 이미지를 컬러 이미지로 변환
 color = cv2.bilateralFilter(img, 9, 300, 300)

 # 컬러 이미지와 가장자리 마스크를 결합
 cartoon = cv2.bitwise_and(color, color, mask=edges)

 # 만화 이미지 표시
 cv2.imshow("Cartoon", cartoon)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
```
## :o: Lens distortion correction
<img src="https://github.com/Mean-Key/MK_CV_CR/blob/main/img/character.png" width="300" height="200"/>  ➡  <img src="https://github.com/Mean-Key/MK_CV_CR/blob/main/img/character-cr.png" width="300" height="200"/>

### 명암 대비가 뚜렷한 이미지   
: 엣지 검출이 뚜렷하게 이루어지고, 색상이 단순화될 때도 자연스럽게 보임.

## :x: 만화같은 느낌이 잘 표현되지 않는 이미지
<img src="https://github.com/Mean-Key/MK_CV_CR/blob/main/img/face.png"/> ➡  <img src="https://github.com/Mean-Key/MK_CV_CR/blob/main/img/face-cr.png">

### 배경이 단순하고 색이 적은 이미지    
: 색이 거의 없으면 `bilateralFilter`의 효과가 미미함.

## :heavy_exclamation_mark: 알고리즘의 한계점
### 1. 조명과 명암에 민감
: 너무 밝거나 어두운 이미지에서는 엣지 검출이 부정확하게 작동할 수 있음.
### 2. 세밀한 디테일 처리 부족 
: 머리카락, 텍스처가 많은 옷, 복잡한 패턴에서는 경계선이 너무 강조되거나 뭉개질 수 있음.
### 3. 고해상도 이미지에서 속도가 느림 
: `bilateralFilter`는 계산량이 많아 고해상도 이미지에서는 처리 속도가 느려짐.
### 4. 일관되지 않은 효과 
: 특정 스타일의 이미지에서는 만화 효과가 잘 나타나지만, 일부 이미지에서는 제대로 표현되지 않을 수 있음.
### 5. 배경이 단순하고 색이 적은 이미지   
: 색이 거의 없으면 `bilateralFilter`의 효과가 미미하며, 엣지 검출도 의미가 없어질 가능성이 있음.
