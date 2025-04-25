import numpy as np
from scipy.signal import find_peaks

# Helper function to check similarity in height
def is_similar(a, b, tolerance=0.1):
    return abs(a - b) / max(a, b) <= tolerance

def detect_head_and_shoulders(df, inverse=False):
    prices = np.ravel(df['Close'].values)
    if inverse:
        prices = -prices

    if prices.size == 0 or np.isnan(prices).all():
        return False, 0, {}

    peaks, _ = find_peaks(prices, distance=5)
    troughs, _ = find_peaks(-prices, distance=5)

    if len(peaks) < 3 or len(troughs) < 2:
        return False, 0, {}

    for i in range(1, len(peaks) - 1):
        ls = peaks[i - 1]
        head = peaks[i]
        rs = peaks[i + 1]

        if not (ls < head < rs):
            continue

        if not (prices[head] > prices[ls] and prices[head] > prices[rs]):
            continue

        if not is_similar(prices[ls], prices[rs], tolerance=0.15):
            continue

        valid_troughs = [t for t in troughs if ls < t < rs]
        if len(valid_troughs) < 2:
            continue

        trough1 = min(valid_troughs, key=lambda x: abs(x - ((ls + head) // 2)))
        trough2 = min(valid_troughs, key=lambda x: abs(x - ((head + rs) // 2)))

        if not (ls < trough1 < head and head < trough2 < rs):
            continue

        neckline_slope = (prices[trough2] - prices[trough1]) / (trough2 - trough1 + 1e-9)
        slope_ok = abs(neckline_slope) < 0.5

        symmetry_score = 1 - abs(prices[ls] - prices[rs]) / prices[head]
        height_ratio_score = min(prices[ls], prices[rs]) / prices[head]
        slope_score = 1 - abs(neckline_slope)

        confidence = (symmetry_score * 0.4 + height_ratio_score * 0.3 + slope_score * 0.3) * 100

        points = {
            'Left Shoulder': ls,
            'Head': head,
            'Right Shoulder': rs,
            'Trough 1': trough1,
            'Trough 2': trough2
        }

        return True, confidence, points

    return False, 0, {}
