import cv2
import numpy as np
import math

def analyze_newton_rings(mask):
    # 轮廓提取
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = []

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5:  # 忽略过小的圆
            diameter = 2 * radius
            diameters.append(diameter)

    diameters.sort()
    return diameters

def analyze_wedge_angle(mask):
    # 边缘提取
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            angles.append(angle_deg)

    return angles
