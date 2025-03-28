import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

def calculate_respiration_rate(rr_intervals, fs, calc_method='fft'):
    """
    rr_intervals: RR 간격 데이터
    fs: 샘플링 주파수
    calc_method: 계산 방법 ('fft', 'welch', 'periodogram' 중 하나)
    """
    if len(rr_intervals) < 3:
        return 0
    
    # 독립 변수 (시간)
    x = np.cumsum(rr_intervals)
    
    # 종속 변수 (RR 간격)
    y = rr_intervals
    
    # 새로운 x 값 (균등 간격)
    x_new = np.linspace(x[0], x[-1], len(rr_intervals))
    
    # 3차 스플라인 보간
    cs = CubicSpline(x, y)
    y_new = cs(x_new)
    
    # 호흡 주파수 범위 필터링 (6-24 breaths/min = 0.1-0.4 Hz)
    nyq = 0.5 * fs
    low = 0.1 / nyq
    high = 0.4 / nyq
    b, a = signal.butter(2, [low, high], btype='band')
    if len(y_new) <= 15:  # padlen이 15인 경우
        return 0  # 충분한 데이터가 없을 때 기본값 반환
    filtered_signal = signal.filtfilt(b, a, y_new)
    
    # 호흡률 계산
    if calc_method == 'fft':
        # FFT 방법
        N = len(filtered_signal)
        fft_values = np.abs(np.fft.fft(filtered_signal))[:N//2]
        freq = np.fft.fftfreq(N, d=1/fs)[:N//2]
        
        # 호흡 범위 내 주파수 찾기
        resp_range = (freq >= 0.1) & (freq <= 0.4)
        if np.any(resp_range):
            dominant_freq = freq[resp_range][np.argmax(fft_values[resp_range])]
            resp_rate = dominant_freq * 60  # Hz to breaths/min
        else:
            resp_rate = 0
            
    elif calc_method == 'welch':
        # Welch 방법
        freq, psd = signal.welch(filtered_signal, fs, nperseg=len(filtered_signal)//2)
        resp_range = (freq >= 0.1) & (freq <= 0.4)
        if np.any(resp_range):
            dominant_freq = freq[resp_range][np.argmax(psd[resp_range])]
            resp_rate = dominant_freq * 60  # Hz to breaths/min
        else:
            resp_rate = 0
            
    elif calc_method == 'periodogram':
        # Periodogram 방법
        freq, psd = signal.periodogram(filtered_signal, fs)
        resp_range = (freq >= 0.1) & (freq <= 0.4)
        if np.any(resp_range):
            dominant_freq = freq[resp_range][np.argmax(psd[resp_range])]
            resp_rate = dominant_freq * 60  # Hz to breaths/min
        else:
            resp_rate = 0
    else:
        raise ValueError("계산 방법은 'fft', 'welch', 또는 'periodogram' 중 하나여야 합니다.")
    
    return resp_rate

def extract_respiration_signal(rppg_signal, fs):
    """
    rPPG 신호에서 직접 호흡 신호를 추출하는 함수.
    베이스라인 변조를 분석하여 호흡 신호를 추출합니다.
    """
    # 신호 필터링 (0.1-0.5 Hz 대역, 호흡 주파수 범위)
    nyquist = 0.5 * fs
    low = 0.1 / nyquist
    high = 0.5 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    
    # 입력 신호 길이 확인
    if len(rppg_signal) < 15:  # filtfilt의 padlen보다 크게 설정
        return np.zeros(len(rppg_signal))
    
    # 신호 필터링
    filtered_signal = signal.filtfilt(b, a, rppg_signal)
    
    return filtered_signal

def calculate_respiration_rate_combined(rppg_signal, rr_intervals, fs):
    """
    여러 방법을 통합하여 호흡률을 계산하는 함수.
    1. RR 간격 기반 방법
    2. 직접 rPPG 신호 분석 방법
    """
    # 방법 1: RR 간격 기반
    resp_rate1 = calculate_respiration_rate(rr_intervals, fs)
    
    # 방법 2: 직접 rPPG 신호 분석
    resp_signal = extract_respiration_signal(rppg_signal, fs)
    
    # FFT 분석
    N = len(resp_signal)
    freq = np.fft.fftfreq(N, d=1/fs)
    fft_values = np.abs(np.fft.fft(resp_signal))[:N//2]
    freq = freq[:N//2]
    
    # 호흡 대역 필터링
    resp_range = (freq >= 0.1) & (freq <= 0.5)
    if np.any(resp_range):
        dominant_freq = freq[resp_range][np.argmax(fft_values[resp_range])]
        resp_rate2 = dominant_freq * 60
    else:
        resp_rate2 = 0
    
    # 두 방법의 결과 통합 (가중 평균 또는 유효한 값 선택)
    if resp_rate1 > 0 and resp_rate2 > 0:
        # 가중 평균 (두 방법 모두 유효한 경우)
        resp_rate = (resp_rate1 + resp_rate2) / 2
    elif resp_rate1 > 0:
        resp_rate = resp_rate1
    elif resp_rate2 > 0:
        resp_rate = resp_rate2
    else:
        resp_rate = 0
    
    return resp_rate



# 테스트 데이터
if __name__ == "__main__":
    fs = 30  # 샘플링 주파수 (FPS)
    
    # 예제 RR 간격 데이터 (밀리초 단위)
    rr_intervals = [800, 810, 790, 820, 805, 795, 810, 800, 815, 805]

    respiration_rate = calculate_respiration_rate(rr_intervals, fs)
    print(f"Estimated Respiration Rate: {respiration_rate:.2f} BPM")
