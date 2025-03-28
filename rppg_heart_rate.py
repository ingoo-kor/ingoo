import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut=0.75, highcut=2.5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    
    # 입력 신호 길이 확인 및 확장
    if len(signal) < 15:  # 최소 필요 길이 증가
        print("Input signal is too short for filtering. Returning original signal...")
        return signal  # 원래 신호 반환
    
    try:
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    except Exception as e:
        print(f"Filtering error: {e}")
        return signal  # 오류 발생 시 원래 신호 반환

def calculate_heart_rate(rppg_signal, fs):
    if len(rppg_signal) < fs * 10: # 최소 5초 데이터 필요
        return 0
    # Bandpass 필터 적용
    filtered_signal = bandpass_filter(rppg_signal, fs)

    # FFT를 사용하여 주파수 도메인 분석
    N = len(filtered_signal)
    freq = np.fft.fftfreq(N, d=1/fs)
    fft_values = np.abs(np.fft.fft(filtered_signal))[:N//2]
    freq = freq[:N//2]

    # 심박수 대역(45~150 BPM)에 해당하는 주파수 선택
    bpm_range = (freq >= 0.75) & (freq <= 2.5)
    if not np.any(bpm_range):  # 유효한 주파수가 없는 경우 처리
        print("No valid frequency found in heart rate range.")
        return 0

    dominant_freq = freq[bpm_range][np.argmax(fft_values[bpm_range])]
    
    # 심박수 계산 (BPM)
    heart_rate_bpm = dominant_freq * 60
    return heart_rate_bpm

# 테스트 데이터 (rPPG 신호 예제)
if __name__ == "__main__":
    fs = 30  # 샘플링 주파수 (FPS)
    time = np.linspace(0, 10, fs * 10)
    
    # 가상의 rPPG 신호 생성 (심박수 약 75 BPM에 해당하는 사인파)
    simulated_rppg_signal = np.sin(2 * np.pi * (75 / 60) * time) + np.random.normal(0, 0.1, len(time))

    heart_rate = calculate_heart_rate(simulated_rppg_signal[:8], fs)  # 짧은 신호로 테스트
    print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
