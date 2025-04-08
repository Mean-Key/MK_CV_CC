# MK's Camera Calibration

## 📌 개요
이 프로그램은 OpenCV를 사용하여 촬영한 영상을 Camera Calibration하는 프로그램입니다.   

## 기능 소개

### Camera calibration 

- **video 선택 - chess_video.mp4 (카메라를 이용해 다양한 시점에서의 체스보드 촬영한 영상)**
```python
video_path = 'chess_video.mp4'
chessboard_size = (9, 6)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []
gray_shape = None
```
- **검은색 테두리 하얀점 표시**
```python
def draw_chessboard_corners(img, corners, radius=6, color=(255, 255, 255)):
    img_vis = img.copy()
    for pt in corners:
        center = tuple(pt[0].astype(int))
        cv.circle(img_vis, center, radius + 2, (0, 0, 0), -1)  
        cv.circle(img_vis, center, radius, color, -1)         
    return img_vis
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cap = cv.VideoCapture(video_path)
frame_interval = 5
frame_idx = 0
```
- **카메라 캘리브레이션**
```python
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
rmse = ret
num_images = len(objpoints)
```
- **결과 출력 코드**
```python
print("\n==============================")
print("## Camera Calibration Results")
print(f"* The number of applied images = {num_images}")
print(f"* RMS error = {rmse:.6f}")
print(f"* Camera matrix (K) = ")
print("[ {:.10f}, 0.0000000000, {:.10f} ]".format(fx, cx))
print("[ 0.0000000000, {:.10f}, {:.10f} ]".format(fy, cy))
print("[ 0.0000000000, 0.0000000000, 1.0000000000 ]")
```

## Lens distortion correction 

- **왜곡 계수 포맷**
```python
dist_list = dist.ravel().tolist()
dist_str = ',\n  '.join([f"{v:.16f}" for v in dist_list])
print("* Distortion coefficient (k1, k2, p1, p2, k3, ...) = ")
print("[ " + dist_str + " ]")
print("==============================\n")
```

- **렌즈 왜곡 보정**
```python
if map1 is None or map2 is None:
    map1, map2 = cv.initUndistortRectifyMap(K, dist, None, K, (w, h), cv.CV_32FC1)

undistorted = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)
```

##  카메라 캘리브레이션 결과
* The number of applied images = 4
* RMS error = 0.566326
* Camera matrix (K) =
[ 1983.8684955170, 0.0000000000, 619.9207220344 ]
[ 0.0000000000, 1987.4953008129, 910.5903161415 ]
[ 0.0000000000, 0.0000000000, 1.0000000000 ]
* Distortion coefficient (k1, k2, p1, p2, k3, ...) =
[ 0.4298971567482212,
  -3.0982602721250032,
  -0.0050665790219284,
  0.0013467451044268,
  7.4240848120337137 ]

## 렌즈 왜곡 보정 결과데모
- 체스판 검은색 테두리 하얀점 표시 (결과 동영상 스크린샷)
<img src="https://github.com/Mean-Key/MK_CV_CC/blob/main/result.png"/>
- 원본 동영상 chess_video.mp4
<img width="50%" src="https://github.com/Mean-Key/MK_CV_CC/blob/main/chess_video.gif"/>
- 카메라 캘리브레이션, 렌즈 왜곡 적용 동영상 undistorted_output.mp4
<img width="50%" src="https://github.com/Mean-Key/MK_CV_CC/blob/main/undistorted_output.gif"/>

