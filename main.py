import cv2
import numpy as np
import matplotlib.pyplot as plt
from face_roi_rppg import select_camera, extract_forehead_roi
from rppg_rr_intervals import extract_rr_intervals
from rppg_heart_rate import calculate_heart_rate
from hrv_stress import calculate_hrv, calculate_stress_index
from respiration_rate import calculate_respiration_rate_combined
from oxygen_saturation import calculate_spo2
from collections import deque
import os
from absl import logging

# 전역 변수 선언
fs = 30 # 샘플링 주파수 (FPS)
cap = None # 카메라 객체

# MediaPipe Face Mesh 초기화
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 시각화 초기화
plt.style.use('dark_background') # 그래프 배경을 어둡게 설정

# 로그 설정: INFO 및 WARNING 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot_graph(ax, signal, title):
    ax.clear()
    ax.plot(signal, color='cyan', linewidth=2)
    ax.set_title(title, color='white')
    ax.set_xticks([])
    ax.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='y', colors='white')

def put_text_with_background(img, text, position, font, font_scale, text_color, thickness):
    """텍스트와 배경 사각형을 함께 표시하는 함수"""
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 배경 사각형 좌표 계산 (약간의 여백 추가)
    x, y = position
    padding = 5
    box_coords = ((x - padding, y + padding), (x + text_width + padding, y - text_height - padding))
    
    # 배경 사각형 그리기
    cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
    
    # 텍스트 그리기
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)

def main():
    global cap
    
    # 카메라 선택 및 초기화
    camera_index = select_camera()
    if camera_index is None:
        print("No valid camera selected. Exiting...")
        return
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Failed to open camera with index {camera_index}. Exiting...")
        return
    
    print(f"Using camera with index {camera_index}.")
    
    # 첫 몇 프레임 버리기 (카메라 안정화)
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            break
    
    # 창 크기 조절 가능하도록 설정
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    
    fs = 30 # 샘플링 주파수 (FPS)
    rppg_signal_buffer = [] # rPPG 신호 버퍼
    
    # Matplotlib 그래프 설정
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.4)
    fig.canvas.manager.set_window_title('Vital Signs')
    
    # 데이터 버퍼 초기화
    rppg_buffer = deque(maxlen=fs*10) # 10초 데이터
    stress_buffer = deque(maxlen=30) # 30개 데이터 포인트
    spo2_buffer = deque(maxlen=30) # 30개 데이터 포인트
    
    # 초기 데이터 설정
    green_signal = np.array([])
    heart_rate, sdnn, rmssd, stress_index, respiration_rate, spo2 = 0, 0, 0, 0, 0, 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 랜드마크 탐지
        results = mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:  # 수정된 부분: face_landmarks -> multi_face_landmarks
            for face_landmarks in results.multi_face_landmarks:  # 수정된 부분
                # 이마 영역 추출
                forehead_roi = extract_forehead_roi(frame.copy(), face_landmarks.landmark)
                
                if forehead_roi is not None and forehead_roi.size > 0:  # 빈 ROI 검사 추가
                    # rPPG 신호 추출 (이마 영역의 RGB 평균값 사용)
                    mean_rgb = np.mean(np.mean(forehead_roi, axis=0), axis=0)
                    rppg_signal_buffer.append(mean_rgb) # 전체 RGB 값 저장
                    
                    if len(rppg_signal_buffer) > fs * 30: # 최대 버퍼 크기 유지 (30초 데이터)
                        rppg_signal_buffer = rppg_signal_buffer[-fs * 30:]
                    
                    # 채널별 신호 분리
                    if len(rppg_signal_buffer) > 0:
                        red_signal = np.array([frame[0] for frame in rppg_signal_buffer])
                        green_signal = np.array([frame[1] for frame in rppg_signal_buffer])
                        blue_signal = np.array([frame[2] for frame in rppg_signal_buffer])
                    
                    # 데이터 수집 상태 표시
                    if len(rppg_signal_buffer) < fs * 10:
                        cv2.putText(frame, "Collecting data...please wait", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # 초기 데이터 수집 중에는 분석 건너뛰기
                        cv2.imshow("Camera Feed", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue
                    
                    try:
                        # 심박수 계산 (녹색 채널 사용)
                        if len(green_signal) > fs * 5: # 최소 5초 데이터 필요
                            heart_rate = calculate_heart_rate(green_signal, fs)
                        else:
                            heart_rate = 0
                        
                        # RR 간격 추출 및 HRV 계산
                        if len(green_signal) > fs * 5:
                            rr_intervals = extract_rr_intervals(green_signal, fs)
                            if len(rr_intervals) > 1:
                                sdnn, rmssd = calculate_hrv(rr_intervals)
                                stress_index = calculate_stress_index(sdnn, rmssd)
                            else:
                                sdnn, rmssd, stress_index = 0, 0, 0
                        else:
                            sdnn, rmssd, stress_index = 0, 0, 0
                    except Exception as e:
                        print(f"Error in processing: {e}")
                        heart_rate, sdnn, rmssd, stress_index = 0, 0, 0, 0
                    
                    # 호흡수 계산
                    try:
                        if len(green_signal) > 30 and len(rr_intervals) > 5: # 충분한 데이터가 있는지 확인
                            respiration_rate = calculate_respiration_rate_combined(green_signal, rr_intervals, fs)
                        else:
                            respiration_rate = 0
                    except ValueError as e:
                        print(f"호흡 분석 오류: {e}")
                        respiration_rate = 0
                    
                    # 산소포화도 계산 (적색 및 녹색 채널 사용)
                    spo2 = calculate_spo2(red_signal, green_signal=green_signal)
                    
                    # 화면 좌측 상단에 텍스트 표시
                    put_text_with_background(frame, f"Heart Rate: {heart_rate:.2f} BPM", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    put_text_with_background(frame, f"SDNN: {sdnn:.2f} ds", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    put_text_with_background(frame, f"RMSSD: {rmssd:.2f} ds", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    put_text_with_background(frame, f"Stress Index: {stress_index:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    put_text_with_background(frame, f"Respiration Rate: {respiration_rate:.2f} BPM", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    put_text_with_background(frame, f"SpO2: {spo2:.1f}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 그래프 업데이트
                    if len(green_signal) > 0:
                        rppg_buffer.extend(green_signal[-min(len(green_signal), fs):])
                    
                    if stress_index > 0:
                        stress_buffer.append(stress_index)
                    
                    if spo2 > 0:
                        spo2_buffer.append(spo2)
                    
                    plot_graph(ax1, list(rppg_buffer), "rPPG Signal")
                    plot_graph(ax2, list(stress_buffer), "Stress Index")
                    plot_graph(ax3, list(spo2_buffer), "SpO2 (%)")
                    
                    plt.pause(0.01)  # 수정된 부분: 구문 오류 제거
        
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키로 종료
            break
    
    cap.release()
    plt.close(fig) # Matplotlib 창 닫기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
