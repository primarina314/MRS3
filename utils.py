# from laptop w/o gpu
import numpy as np
import cv2
import time
import os
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

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




print(cv2.__version__)


