# from laptop w/o gpu
import numpy as np
import cv2
import time
import os
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image


#####################################
# TODO
# Select ROI
# Mode 2+ 개 나눠서 옵션
# - rect
# - poly
# - curve
# - etc
#####################################

class ROI_mode(Enum):
    Rectengle = 0
    Polygon = 1

def crop_by_drag(image_path, mode=ROI_mode.Rectengle):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # ROI(관심영역) 선택 창 띄우기 (마우스로 드래그)
    x, y, w, h = cv2.selectROI("이미지에서 영역을 드래그하세요", img, showCrosshair=True, fromCenter=False)

    # ROI가 정상적으로 선택된 경우에만 진행
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        cv2.imshow("선택된 영역", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("영역이 선택되지 않았습니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# crop_by_drag('Lenna_(test_image).png')

# 마우스 콜백 함수
drawing = False
points = []

def draw_polygon(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
        drawing = False  # 다각형 닫기

def select_polygon_roi(image_path):
    global drawing, points
    img = cv2.imread(image_path)
    clone = img.copy()
    cv2.namedWindow("이미지에서 다각형 ROI 지정")
    cv2.setMouseCallback("이미지에서 다각형 ROI 지정", draw_polygon)

    drawing = True
    while drawing:
        temp = clone.copy()
        if len(points) > 0:
            cv2.polylines(temp, [np.array(points)], False, (0,255,0), 2)
            for pt in points:
                cv2.circle(temp, pt, 3, (0,0,255), -1)
        cv2.imshow("이미지에서 다각형 ROI 지정", temp)
        key = cv2.waitKey(1)
        if key == 27:  # ESC로 취소
            points = []
            break
        if key == ord('s') and len(points) >= 3:  # 's'로 저장
            drawing = False

    if len(points) >= 3:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        roi = cv2.bitwise_and(img, img, mask=mask)
        # 다각형의 bounding box로 crop
        pts = np.array(points)
        x, y, w, h = cv2.boundingRect(pts)
        cropped = roi[y:y+h, x:x+w]
        cv2.imshow("다각형 ROI", cropped)
        cv2.waitKey(0)
    else:
        print("3개 이상의 꼭짓점이 필요합니다.")

    cv2.destroyAllWindows()
    return cropped, (y, y+h, x, x+w)

cropped_test_img, loc = select_polygon_roi('Lenna_(test_image).png')
# 사용 예시
# cv2.imshow('cropped', select_polygon_roi('Lenna_(test_image).png'))



################################
# upscale
################################

def upscale_img(image_path, scaler):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    if scaler not in [2, 3, 4]:
        print(f"Invalid scaler value: {scaler}. Must be 2, 3 or 4.")
        return None
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-enabled GPU found.")
        return None

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel(f'models/EDSR_x{scaler}.pb')
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

    # gpu acceleration
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    sr.setModel('edsr', scaler)

    try:
        result = sr.upsample(img)
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return None

    return result


def downscale_img(image_path, scaler):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    new_h, new_w = img.shape[0]//scaler, img.shape[1]//scaler

    result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return result

print(cv2.__version__)


