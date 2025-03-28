import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# Bandpass 필터 생성 함수 (주파수 대역: 0.75~2.5 Hz)
def bandpass_filter(signal, fs, lowcut=0.75, highcut=2.5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')  # 1차 Butterworth 필터
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# RR 간격 추출 함수
def extract_rr_intervals(rppg_signal, fs):
    peaks, _ = find_peaks(rppg_signal, distance=fs*0.6)  # 최소 0.6초 간격 (100 BPM 기준)
    rr_intervals = np.diff(peaks) / fs * 1000  # 밀리초(ms) 단위로 변환
    return rr_intervals

# HRV 계산 함수
def calculate_hrv(rr_intervals):
    sdnn = np.std(rr_intervals)
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs**2))
    return sdnn, rmssd

# 스트레스 지수 계산 함수
def calculate_stress_index(sdnn, rmssd):
    if sdnn == 0 or rmssd == 0:
        return float('inf')
    stress_index = (1 / rmssd) * (100 / sdnn)
    return stress_index

# 시각화 초기화
plt.style.use('dark_background')  # 그래프 배경을 어둡게 설정

def plot_graph(ax, signal):
    ax.clear()
    ax.plot(signal, color='cyan', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 축 및 배경 제거

# 실시간 처리 및 시각화
def main():
    cap = cv2.VideoCapture(0)  # 카메라 열기
    fs = 30  # 샘플링 주파수 (FPS)
    rppg_signal_buffer = []  # rPPG 신호 버퍼
    
    # Matplotlib 그래프 설정
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.canvas.manager.window.setGeometry(800, 0, 400, 200)  # 그래프 창 위치 설정

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB 변환 및 rPPG 신호 임의 생성 (실제 환경에서는 추출된 rPPG 신호 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simulated_rppg_signal = np.sin(2 * np.pi * (75 / 60) * np.linspace(0, len(rgb_frame), fs)) + np.random.normal(0, 0.1)

        # rPPG 신호 버퍼 업데이트
        rppg_signal_buffer.extend(simulated_rppg_signal)
        if len(rppg_signal_buffer) > fs * 10:  # 최대 버퍼 크기 유지 (10초 데이터)
            rppg_signal_buffer = rppg_signal_buffer[-fs * 10:]

        # RR 간격 추출 및 HRV 계산
        rr_intervals = extract_rr_intervals(rppg_signal_buffer, fs)
        if len(rr_intervals) > 1:
            sdnn, rmssd = calculate_hrv(rr_intervals)
            stress_index = calculate_stress_index(sdnn, rmssd)

            # 화면 좌측 상단에 텍스트 표시
            cv2.putText(frame, f"HRV Metrics:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"SDNN: {sdnn:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"RMSSD: {rmssd:.2f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Stress Index: {stress_index:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 그래프 업데이트 및 표시
        plot_graph(ax=ax, signal=rppg_signal_buffer[-fs*5:])  # 최근 데이터만 표시 (5초간 데이터)
        plt.pause(0.01)

        # 화면 출력
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):   # 'q' 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
