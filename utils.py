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
# drawing = False
# points = []

# def draw_polygon(event, x, y, flags, param):
#     global drawing, points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#     elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
#         drawing = False  # 다각형 닫기

# def select_polygon_roi(image_path):
#     global drawing, points
#     img = cv2.imread(image_path)
#     clone = img.copy()
#     cv2.namedWindow("이미지에서 다각형 ROI 지정")
#     cv2.setMouseCallback("이미지에서 다각형 ROI 지정", draw_polygon)

#     drawing = True
#     while drawing:
#         temp = clone.copy()
#         if len(points) > 0:
#             cv2.polylines(temp, [np.array(points)], False, (0,255,0), 2)
#             for pt in points:
#                 cv2.circle(temp, pt, 3, (0,0,255), -1)
#         cv2.imshow("이미지에서 다각형 ROI 지정", temp)
#         key = cv2.waitKey(1)
#         if key == 27:  # ESC로 취소
#             points = []
#             break
#         if key == ord('s') and len(points) >= 3:  # 's'로 저장
#             drawing = False

#     if len(points) >= 3:
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         cv2.fillPoly(mask, [np.array(points)], 255)
#         roi = cv2.bitwise_and(img, img, mask=mask)
#         # 다각형의 bounding box로 crop
#         pts = np.array(points)
#         x, y, w, h = cv2.boundingRect(pts)
#         cropped = roi[y:y+h, x:x+w]
#         cv2.imshow("다각형 ROI", cropped)
#         cv2.waitKey(0)
#     else:
#         print("3개 이상의 꼭짓점이 필요합니다.")

#     cv2.destroyAllWindows()
#     return cropped, (y, y+h, x, x+w)

# 사용 예시
# cv2.imshow('cropped', select_polygon_roi('Lenna_(test_image).png'))

############################
# 사각형 원본 저장 - 좌표 기준
############################

def specify_part_of_original(image_path, r_from, r_to, c_from, c_to):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # 클릭, 드래그 등으로 지정
    # r_from = 100
    # r_to = 300
    # c_from = 200
    # c_to = 400

    return img[r_from:r_to, c_from:c_to]



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

################################
# combine two images
################################


def combine_images(A, B):
    """
    A: (H, W, 3) shape의 넘파이 배열 (검은색 픽셀 기준)
    B: (H, W, 3) shape의 넘파이 배열 (A의 검은 픽셀 대체용)
    """
    # 1. A 이미지에서 검은색 픽셀 마스크 생성
    black_mask = np.all(A == [0, 0, 0], axis=2)
    
    # 2. 3채널에 적용 가능하도록 차원 확장
    mask_3d = black_mask[:, :, np.newaxis]
    
    # 3. 조건에 따라 픽셀 선택
    return np.where(mask_3d, B, A)


################################
# rename images for convinience
################################

def rename_images_by_resolution(folder_path):
    # 해상도별로 파일 개수를 기록할 딕셔너리
    resolution_count = defaultdict(int)

    # 폴더 내 모든 png 파일 목록
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                resolution = f"{width}x{height}"

                # 해당 해상도 파일 개수 증가
                resolution_count[resolution] += 1
                count = resolution_count[resolution]

                # 새 파일명 생성
                if count == 1:
                    new_name = f"{resolution}.png"
                else:
                    new_name = f"{resolution}-{count-1}.png"

                new_path = os.path.join(folder_path, new_name)

                # 파일명 변경
                os.rename(file_path, new_path)
                print(f"Renamed '{file_name}' to '{new_name}'")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# 사용 예시 (폴더 경로를 원하는 경로로 바꿔서 사용)
# rename_images_by_resolution('sample-images-png')


#########################
# 이미지 내의 검은색(0 0 0) 픽셀 비율
#########################

def black_pixel_ratio(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    # (H, W, 3) 배열에서 [0,0,0]인 픽셀 찾기
    black_mask = np.all(img == [0, 0, 0], axis=2)
    # axis=2: channel -> (h, w, c) 에서 c 가 사라진 (h, w) 로 리턴 shape
    black_count = np.sum(black_mask)
    total_pixels = img.shape[0] * img.shape[1]
    ratio = black_count / total_pixels

    print(f"검은 픽셀 개수: {black_count}")
    print(f"전체 픽셀 개수: {total_pixels}")
    print(f"검은 픽셀 비율: {ratio:.4%}")
    return ratio

# black_pixel_ratio('cropped.png')

# img_path = 'sample-images-png/1920x1080.png'
# original_part, original_part_loc = select_polygon_roi(img_path)

print(cv2.__version__)


