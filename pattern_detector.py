import numpy as np
from scipy.signal import find_peaks

# Helper function to check similarity in height
def is_similar(a, b, tolerance=0.1):
    return bool(abs(a - b) / max(a, b) <= tolerance)

def detect_head_and_shoulders(df, inverse=False):
    prices = np.ravel(df['Close'].values)
    if inverse:
        prices = -prices

    if prices.size == 0 or np.isnan(prices).all():
        return False, 0, {}

    peaks, _ = find_peaks(prices, distance=5)
    troughs, _ = find_peaks(-prices, distance=5)

    # Ensure peaks and troughs are 1D arrays of integers
    if len(peaks) < 3 or len(troughs) < 2:
        return False, 0, {}

    peaks = peaks.astype(int)
    troughs = troughs.astype(int)

    for i in range(1, len(peaks) - 1):
        try:
            ls = int(peaks[i - 1])
            head = int(peaks[i])
            rs = int(peaks[i + 1])

            if not (ls < head and head < rs):
                continue

            lhs = float(prices[ls])
            hd = float(prices[head])
            rhs = float(prices[rs])

            if not (hd > lhs and hd > rhs):
                continue

            if not is_similar(lhs, rhs, tolerance=0.15):
                continue

            valid_troughs = [int(t) for t in troughs if ls < t < rs]
            if len(valid_troughs) < 2:
                continue

            trough1 = min(valid_troughs, key=lambda x: abs(x - ((ls + head) // 2)))
            trough2 = min(valid_troughs, key=lambda x: abs(x - ((head + rs) // 2)))

            if not (ls < trough1 < head and head < trough2 < rs):
                continue

            neckline_slope = (prices[trough2] - prices[trough1]) / (trough2 - trough1 + 1e-9)

            symmetry_score = float(1 - abs(lhs - rhs) / hd)
            height_ratio_score = float(min(lhs, rhs) / hd)
            slope_score = float(1 - abs(neckline_slope))

            confidence = (symmetry_score * 0.4 + height_ratio_score * 0.3 + slope_score * 0.3) * 100

            points = {
                'Left Shoulder': ls,
                'Head': head,
                'Right Shoulder': rs,
                'Trough 1': trough1,
                'Trough 2': trough2
            }

            return True, confidence, points

        except Exception as e:
            print(f"❌ Error processing peaks at index {i}: {e}")
            continue

    return False, 0, {}
