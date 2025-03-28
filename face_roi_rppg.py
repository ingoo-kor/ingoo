import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import detrend
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# ROI 추출 함수 (이마 영역)
def extract_forehead_roi(image, landmarks):
    forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]  # 이마 부위의 랜드마크 인덱스
    h, w, _ = image.shape
    forehead_points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in forehead_indices])
    x_min, y_min = np.min(forehead_points[:, 0]), np.min(forehead_points[:, 1])
    x_max, y_max = np.max(forehead_points[:, 0]), np.max(forehead_points[:, 1])
    roi = image[y_min:y_max, x_min:x_max]
    return roi

# ROI 추출 함수 (뺨 영역)
def extract_cheek_roi(image, landmarks):
    left_cheek_indices = [234, 93, 132, 58, 172]   # 왼쪽 뺨 부위 랜드마크
    right_cheek_indices = [454, 323, 361, 288, 397] # 오른쪽 뺨 부위 랜드마크
    h, w, _ = image.shape

    left_cheek_points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in left_cheek_indices])
    left_x_min, left_y_min = np.min(left_cheek_points[:, 0]), np.min(left_cheek_points[:, 1])
    left_x_max, left_y_max = np.max(left_cheek_points[:, 0]), np.max(left_cheek_points[:, 1])
    left_roi = image[left_y_min:left_y_max, left_x_min:left_x_max]

    right_cheek_points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in right_cheek_indices])
    right_x_min, right_y_min = np.min(right_cheek_points[:, 0]), np.min(right_cheek_points[:, 1])
    right_x_max, right_y_max = np.max(right_cheek_points[:, 0]), np.max(right_cheek_points[:, 1])
    right_roi = image[right_y_min:right_y_max, right_x_min:right_x_max]

    return left_roi, right_roi

# rPPG 신호 계산 함수 (RGB 평균값 기반)
def calculate_rppg_signal(roi_frames):
    rgb_signals = []
    for roi in roi_frames:
        mean_rgb = np.mean(np.mean(roi, axis=0), axis=0) if roi is not None else [0, 0, 0]
        rgb_signals.append(mean_rgb)
    
    rgb_signals = np.array(rgb_signals)
    
    # 각 채널별로 detrend 적용
    detrended_signals = np.zeros_like(rgb_signals)
    if len(rgb_signals) > 0:
        for i in range(rgb_signals.shape[1]):
            detrended_signals[:, i] = detrend(rgb_signals[:, i])
    
    return detrended_signals  # [R, G, B] 채널 모두 반환


# 카메라 탐지 및 선택 함수 추가
def list_available_cameras():
    """
    시스템에 연결된 모든 카메라를 나열하는 함수.
    """
    index = 0
    available_cameras = []
    
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_cameras.append(index)
        cap.release()
        index += 1
    
    if not available_cameras:
        print("No cameras detected.")
        return None
    
    print("Available Cameras:")
    for cam_index in available_cameras:
        print(f"Camera Index: {cam_index}")
    
    return available_cameras

def select_camera():
    """
    사용자가 선택한 카메라의 인덱스를 반환하는 함수.
    """
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        return None
    
    while True:
        try:
            camera_index = int(input("Enter the index of the camera you want to use: "))
            if camera_index in available_cameras:
                return camera_index
            else:
                print("Invalid index. Please select a valid camera index.")
        except ValueError:
            print("Please enter a valid integer.")

