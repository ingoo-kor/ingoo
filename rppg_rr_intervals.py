import numpy as np
from scipy.signal import find_peaks

# RR 간격 추출 함수
def extract_rr_intervals(rppg_signal, fs):
    """
    rPPG 신호에서 RR 간격을 추출하는 함수.
    - rPPG 신호: 심박수를 포함한 신호
    - fs: 샘플링 주파수 (FPS)
    """
    # 1. 심박수 피크 감지
    peaks, _ = find_peaks(rppg_signal, distance=fs*0.6) # 최소 0.6초 간격 (100 BPM 기준)
    
    # 2. 피크 간 시간 차이 계산 (RR 간격)
    rr_intervals = (np.diff(peaks) / fs) * 10 # 초(ds) 단위로 변환 - 수정된 부분
    
    return rr_intervals

# 테스트 데이터 (rPPG 신호 예제)
if __name__ == "__main__":
    # 샘플 데이터: rPPG 신호 (임의의 데이터로 테스트 가능)
    fs = 30  # 샘플링 주파수 (FPS)
    time = np.linspace(0, 10, fs * 10)  # 시간 축 (10초간 데이터)
    
    # 가상의 rPPG 신호 생성 (심박수 약 75 BPM에 해당하는 사인파)
    simulated_rppg_signal = np.sin(2 * np.pi * (75 / 60) * time) + np.random.normal(0, 0.1, len(time))

    # RR 간격 추출
    rr_intervals = extract_rr_intervals(simulated_rppg_signal, fs)
    print(f"RR Intervals: {rr_intervals}")


# 추출된 RR간격 데이터를 hrv_stress.py에 입력해 HRV와 스트레스 지수 계산해야함