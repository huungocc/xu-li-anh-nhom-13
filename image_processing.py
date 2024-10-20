import cv2
import numpy as np


def colorgrad(f, mask_type='sobel', T=0):
    if len(f.shape) != 3 or f.shape[2] != 3:
        raise ValueError('Input image must be RGB')

    if mask_type == 'sobel':
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    elif mask_type == 'prewitt':
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    else:
        raise ValueError('Invalid mask type. Use "sobel" or "prewitt".')

    f = f.astype(np.float32) / 255.0

    Rx = cv2.filter2D(f[:, :, 0], -1, kx)
    Ry = cv2.filter2D(f[:, :, 0], -1, ky)
    Gx = cv2.filter2D(f[:, :, 1], -1, kx)
    Gy = cv2.filter2D(f[:, :, 1], -1, ky)
    Bx = cv2.filter2D(f[:, :, 2], -1, kx)
    By = cv2.filter2D(f[:, :, 2], -1, ky)

    gxx = Rx ** 2 + Gx ** 2 + Bx ** 2
    gyy = Ry ** 2 + Gy ** 2 + By ** 2
    gxy = Rx * Ry + Gx * Gy + Bx * By

    A = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    G1 = 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * A) + 2 * gxy * np.sin(2 * A))

    A = A + np.pi / 2
    G2 = 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * A) + 2 * gxy * np.sin(2 * A))

    # Xử lý các giá trị âm hoặc NaN trước khi tính căn bậc hai
    G1 = np.clip(G1, 0, None)  # Đảm bảo G1 không âm
    G2 = np.clip(G2, 0, None)  # Đảm bảo G2 không âm

    G1 = np.sqrt(G1)
    G2 = np.sqrt(G2)

    VG = np.maximum(G1, G2)

    # Chuẩn hóa VG trong khoảng [0, 1]
    VG_min, VG_max = VG.min(), VG.max()
    if VG_max > VG_min:
        VG = (VG - VG_min) / (VG_max - VG_min)
    else:
        VG = np.zeros_like(VG)  # Nếu VG_max == VG_min, set tất cả về 0

    RG = np.sqrt(np.clip(Rx ** 2 + Ry ** 2, 0, None))
    GG = np.sqrt(np.clip(Gx ** 2 + Gy ** 2, 0, None))
    BG = np.sqrt(np.clip(Bx ** 2 + By ** 2, 0, None))

    PPG = RG + GG + BG

    # Chuẩn hóa PPG trong khoảng [0, 1]
    PPG_min, PPG_max = PPG.min(), PPG.max()
    if PPG_max > PPG_min:
        PPG = (PPG - PPG_min) / (PPG_max - PPG_min)
    else:
        PPG = np.zeros_like(PPG)  # Nếu PPG_max == PPG_min, set tất cả về 0

    if T > 0:
        VG = np.where(VG > T, VG, 0)
        PPG = np.where(PPG > T, PPG, 0)

    return VG, A, PPG