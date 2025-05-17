
import sys
import cv2
from analysis.circle_fitting import detect_newton_rings_opencv

def detect_and_draw(img_path):
    rings = detect_newton_rings_opencv(img_path)
    img = cv2.imread(img_path)

    if not rings:
        print("❌ 未检测到任何圆")
        return

    print(f"✅ 检测到 {len(rings)} 个圆:")
    for i, (x, y, r) in enumerate(rings):
        print(f"  - 圆{i+1}: 中心=({x}, {y}), 半径={r}")
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

    cv2.imshow("Detected Newton Rings", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_by_opencv.py <image_path>")
    else:
        detect_and_draw(sys.argv[1])
