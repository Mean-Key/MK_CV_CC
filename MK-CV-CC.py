import cv2 as cv
import numpy as np

# === 설정 ===
video_path = 'chess_video.mp4'
chessboard_size = (9, 6)

# === 체스보드 3D 좌표 준비 ===
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []
gray_shape = None

# === 꼭짓점 표시 함수 (흰 점 + 검정 테두리)
def draw_chessboard_corners(img, corners, radius=6, color=(255, 255, 255)):
    img_vis = img.copy()
    for pt in corners:
        center = tuple(pt[0].astype(int))
        cv.circle(img_vis, center, radius + 2, (0, 0, 0), -1)  # 검정 테두리
        cv.circle(img_vis, center, radius, color, -1)         # 흰색 점
    return img_vis

# === 체스보드 감지 및 포인트 수집 ===
cap = cv.VideoCapture(video_path)
frame_interval = 5
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        found, corners = cv.findChessboardCorners(gray, chessboard_size)
        if found:
            corners_sub = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners_sub)

    frame_idx += 1

cap.release()

if len(objpoints) == 0:
    print("❌ 체스보드를 한 번도 인식하지 못했습니다.")
    exit()

# === 카메라 캘리브레이션 ===
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
rmse = ret
num_images = len(objpoints)

# === 결과 출력 ===
print("\n==============================")
print("## Camera Calibration Results")
print(f"* The number of applied images = {num_images}")
print(f"* RMS error = {rmse:.6f}")
print(f"* Camera matrix (K) = ")
print("[ {:.10f}, 0.0000000000, {:.10f} ]".format(fx, cx))
print("[ 0.0000000000, {:.10f}, {:.10f} ]".format(fy, cy))
print("[ 0.0000000000, 0.0000000000, 1.0000000000 ]")

# 왜곡 계수 포맷
dist_list = dist.ravel().tolist()
dist_str = ',\n  '.join([f"{v:.16f}" for v in dist_list])
print("* Distortion coefficient (k1, k2, p1, p2, k3, ...) = ")
print("[ " + dist_str + " ]")
print("==============================\n")

# === 보정된 영상 보기 + 저장 ===
cap = cv.VideoCapture(video_path)
map1, map2 = None, None
frame_idx = 0

# === 영상 저장 설정 ===
output_path = "undistorted_output.mp4"
fps = 30
fourcc = cv.VideoWriter_fourcc(*'mp4v')
ret, sample_frame = cap.read()
h, w = sample_frame.shape[:2]
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
out = cv.VideoWriter(output_path, fourcc, fps, (w, h))

print("보정된 영상 보기 및 저장 중... (ESC 키로 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if map1 is None or map2 is None:
        map1, map2 = cv.initUndistortRectifyMap(K, dist, None, K, (w, h), cv.CV_32FC1)

    undistorted = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)

    # 꼭짓점 다시 찾고 표시
    gray = cv.cvtColor(undistorted, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, chessboard_size)
    if found:
        undistorted = draw_chessboard_corners(undistorted, corners)

    # 저장
    out.write(undistorted)

    # 보기용 리사이즈
    resized = cv.resize(undistorted, None, fx=0.3, fy=0.3)
    cv.imshow('Undistorted with Corners', resized)

    if cv.waitKey(30) == 27:
        break

    frame_idx += 1

cap.release()
out.release()
cv.destroyAllWindows()
