# MK's Camera Calibration

## 📌 개요
이 프로그램은 OpenCV를 사용하여 촬영한 영상을 Camera Calibration하는 프로그램입니다.   

## 기능 소개
- **Camera calibration**
```python
# video 선택 - chess_video.mp4 (카메라를 이용해 다양한 시점에서의 체스보드 촬영한 영상)
video_path = 'chess_video.mp4'
chessboard_size = (9, 6)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []
gray_shape = None

# 검은색 테두리 하얀점 표시
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
## 결과출력
```python
# === 결과 출력 ===
print("\n==============================")
print("## Camera Calibration Results")
print(f"* The number of applied images = {num_images}")
print(f"* RMS error = {rmse:.6f}")
print(f"* Camera matrix (K) = ")
print("[ {:.10f}, 0.0000000000, {:.10f} ]".format(fx, cx))
print("[ 0.0000000000, {:.10f}, {:.10f} ]".format(fy, cy))
print("[ 0.0000000000, 0.0000000000, 1.0000000000 ]")
```

# Lens distortion correction 
```python
# 왜곡 계수 포맷
dist_list = dist.ravel().tolist()
dist_str = ',\n  '.join([f"{v:.16f}" for v in dist_list])
print("* Distortion coefficient (k1, k2, p1, p2, k3, ...) = ")
print("[ " + dist_str + " ]")
print("==============================\n")
```
