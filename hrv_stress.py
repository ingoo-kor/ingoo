import numpy as np

# HRV 계산 함수
def calculate_hrv(rr_intervals):
    """
    HRV를 계산하는 함수.
    - SDNN: NN 간격의 표준편차
    - RMSSD: 연속적인 NN 간격 차이의 제곱 평균의 제곱근
    """
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_intervals)

    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs**2))

    return sdnn, rmssd

# 스트레스 지수 계산 함수
def calculate_stress_index(sdnn, rmssd):
    """
    스트레스 지수를 계산하는 함수.
    - 스트레스 지수 공식: (1 / RMSSD) * (100 / SDNN)
    높은 RMSSD는 낮은 스트레스를, 낮은 SDNN은 높은 스트레스를 나타냄.
    """
    if sdnn < 1e-6 or rmssd < 1e-6: # 매우 작은 값을 방지
        return float('inf') # 무한대 반환
    
    stress_index = (1 / rmssd) * (100 / sdnn) / 10
    
    return stress_index

# 테스트 데이터 (RR 간격 예제, 단위: 밀리초)
if __name__ == "__main__":
    # RR 간격 데이터 (예제)
    rr_intervals = [800, 810, 790, 820, 805, 795, 810, 800, 815, 805]  # 단위: ms

    # HRV 계산
    sdnn, rmssd = calculate_hrv(rr_intervals)
    print(f"SDNN: {sdnn:.2f} ms")
    print(f"RMSSD: {rmssd:.2f} ms")

    # 스트레스 지수 계산
    stress_index = calculate_stress_index(sdnn, rmssd)
    print(f"Stress Index: {stress_index:.2f}")
