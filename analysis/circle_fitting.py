
import cv2
import numpy as np

def preprocess_for_circle_detection(img_path, sigma=1.0):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return img, thresh

def detect_newton_rings_opencv(img_path, min_radius=20, max_radius=200):
    img, thresh = preprocess_for_circle_detection(img_path)
    edges = cv2.Canny(thresh, 50, 150)

    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=30,
        minRadius=min_radius, maxRadius=max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        valid_circles = []
        for x, y, r in circles[0, :]:
            if abs(x - center_x) < 50 and abs(y - center_y) < 50:
                valid_circles.append((x, y, r))

        valid_circles.sort(key=lambda c: c[2])
        return valid_circles
    else:
        return []

def fit_circle_least_squares(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    A = np.column_stack((x - x_mean, y - y_mean))
    b = (x**2 + y**2) - (x_mean**2 + y_mean**2)
    params, _ = np.linalg.lstsq(A, b, rcond=None)[0:2]
    a, b = params
    radius = np.sqrt(a**2 + b**2 + x_mean**2 + y_mean**2 - 2*(a*x_mean + b*y_mean))
    return (x_mean + a, y_mean + b), radius
