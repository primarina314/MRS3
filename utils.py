# from laptop w/o gpu
import numpy as np
import cv2
import time
import os
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import configparser

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


# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_InterpolationFlags.html
# https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
# https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121a94cc44aa86159abcff4683ec6841b097

class MRS3_mode(Enum):
    edsr = -1
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    INTER_LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    INTER_NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    INTER_MAX = cv2.INTER_MAX
    WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP
    WARP_RELATIVE_MAP = cv2.WARP_RELATIVE_MAP

def select_rectangle_roi(image_path):
    """
    input_path: 추출할 이미지 경로
    return: roi ndarray, (from_y, to_y, from_x, to_x)
    """

    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return None, None

    # ROI(관심영역) 선택 창 띄우기 (마우스로 드래그)
    x, y, w, h = cv2.selectROI("select part to remain by dragging", img, showCrosshair=True, fromCenter=False)

    # ROI가 정상적으로 선택된 경우에만 진행
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        cv2.imshow("selected part", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return roi, (y, y+h, x, x+w)
    else:
        print("None of part selected")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("none")
        return None, None


# mousecallback
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
    cv2.namedWindow("indicate polygon in img")
    cv2.setMouseCallback("indicate polygon in img", draw_polygon)

    drawing = True
    while drawing:
        temp = clone.copy()
        if len(points) > 0:
            cv2.polylines(temp, [np.array(points)], False, (0,255,0), 2)
            for pt in points:
                cv2.circle(temp, pt, 3, (0,0,255), -1)
        cv2.imshow("indicate polygon in img", temp)
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
        cv2.imshow("polygon ROI", cropped)
        cv2.waitKey(0)
    else:
        print("3개 이상의 꼭짓점이 필요합니다.")
        return None, None

    cv2.destroyAllWindows()
    return cropped, (y, y+h, x, x+w)

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
# TODO: 메모리 한계 넘어가는 큰 이미지는 분할해서. 분할된 경계가 조금씩 겹치도록 한 후, 여기에도 자연스럽게 blending
################################

def upscale_by_edsr(image_path, scaler):
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

def upscale_by_resize(image_path, scaler, interpolation = cv2.INTER_CUBIC):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    h, w = img.shape[0] * scaler, img.shape[1] * scaler
    result = cv2.resize(img, (w, h), interpolation=interpolation)
    return result

################################
# downscale
################################
def downscale_img(image_path, scaler, interpolation = cv2.INTER_AREA):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    new_h, new_w = img.shape[0]//scaler, img.shape[1]//scaler

    result = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
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



# mrs3 적용 후 파일저장 경로/이름
roi_path_prefix = 'roi' # png
downscaled_path = 'downscaled' # png
config_path = 'config' # ini

# mrs3 mode, select roi mode
def mrs3_compress(img_path, output_path, scaler, roi_mode, interpolation=cv2.INTER_AREA):
    """
    edsr 기반으로 mrs3
    img_path: mrs3 적용할 이미지 경로
    output_path: 결과 저장할 폴더 경로
    scaler: 이미지 downscale 배율
    """

    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return

    # TODO: 타겟 개수 조정할 수 있도록 추가

    # 이미지 축소 및 roi 저장
    if roi_mode == ROI_mode.Rectengle:
        roi, loc = select_rectangle_roi(img_path)
    
    if roi_mode == ROI_mode.Polygon:
        roi, loc = select_polygon_roi(img_path)

    # TODO: 다양한 interpolation 비교 및 복원 비교
    downscaled = downscale_img(img_path, scaler, interpolation=interpolation)


    # 메타데이터 ini 에 저장
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'SCALER': f'{scaler}',
        'NUMBER_OF_TARGETS': f'1'
    }
    config['0'] = {
        'Y_FROM': f'{loc[0]}',
        'Y_TO': f'{loc[1]}',
        'X_FROM': f'{loc[2]}',
        'X_TO': f'{loc[3]}'
    }
    # TODO: 타겟 넘버링하여 각자 위치 정보 저장

    # 결과저장 폴더 없을 때 새로 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # TODO: 타겟 이미지 저장 경로 리스트로 0부터 n-1 까지 순차저장
    cv2.imwrite(f'{output_path}/{downscaled_path}.png', downscaled)
    cv2.imwrite(f'{output_path}/{roi_path_prefix}{0}.png', roi)
    with open(f'{output_path}/{config_path}.ini', 'w') as configfile:
        config.write(configfile)

    filesize_bef = os.path.getsize(img_path)
    filesize_downscaled = os.path.getsize(f'{output_path}/{downscaled_path}.png')
    filesize_roi = os.path.getsize(f'{output_path}/{roi_path_prefix}{0}.png')
    filesize_config = os.path.getsize(f'{output_path}/{config_path}.ini')

    print(f'original file: {filesize_bef}')
    print(f'downscaled filesize: {filesize_downscaled}')
    print(f'roi filesize: {filesize_roi}')
    print(f'config filesize: {filesize_config}')

    # 파일 사이즈 압축률 print
    print(f'compression ratio: {(filesize_downscaled + filesize_roi + filesize_config) / filesize_bef}')

    return

def mrs3_restore(input_path, mrs3_mode, output_path=""):
    """
    mrs3 처리한 후, 이미지 복원
    input_path: mrs3 적용한 폴더 경로 - 나중에 폴더말고 하나의 파일형식에 저장하도록 수정하는게 좋을듯.
    """

    if not os.path.exists(f'{input_path}/{downscaled_path}.png'):
        print(f"Error loading image: {input_path}/{downscaled_path}.png")
        return

    if not os.path.exists(f'{input_path}/{config_path}.ini'):
        print(f'Error loading config: {input_path}/{config_path}.ini')
        return
    
    if not os.path.exists(f'{input_path}/{roi_path_prefix}{0}.png'):
        print(f'Error loading image: {input_path}/{roi_path_prefix}{0}.png')
        return

    config = configparser.ConfigParser()
    config.read(f'{input_path}/{config_path}.ini')


    y_from, y_to, x_from, x_to = int(config['0']['Y_FROM']), int(config['0']['Y_TO']), int(config['0']['X_FROM']), int(config['0']['X_TO'])
    scaler = int(config['DEFAULT']['SCALER'])

    if mrs3_mode == MRS3_mode.edsr:
        restored = upscale_by_edsr(f'{input_path}/{downscaled_path}.png', scaler=scaler)
    else:
        restored = upscale_by_resize('{input_path}/{downscaled_path}.png', scaler=scaler, interpolation=mrs3_mode)
    

    # TODO: 다수 roi 반영 0 ~ n-1
    roi = cv2.imread(f'{input_path}/{roi_path_prefix}{0}.png')    
    combined_roi = combine_images(roi, restored[y_from:y_to, x_from:x_to])
    restored[y_from:y_to, x_from:x_to] = combined_roi

    cv2.imshow('restored img', restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path != "":
        cv2.imwrite(output_path, restored)
    return


# img_path = 'sample-images-png/1920x1080.png'
# original_part, original_part_loc = select_polygon_roi(img_path)



# print("interpolation flags")
# print(cv2.INTER_NEAREST)
# print(cv2.INTER_LINEAR)
# print(cv2.INTER_CUBIC)
# print(cv2.INTER_AREA)
# print(cv2.INTER_LANCZOS4)
# print(cv2.INTER_LINEAR_EXACT)
# print(cv2.INTER_NEAREST_EXACT)
# print(cv2.INTER_MAX)
# print(cv2.WARP_FILL_OUTLIERS)
# print(cv2.WARP_INVERSE_MAP)
# print(cv2.WARP_RELATIVE_MAP)

# print("interpolation masks")
# print(cv2.INTER_BITS)
# print(cv2.INTER_BITS2)
# print(cv2.INTER_TAB_SIZE)
# print(cv2.INTER_TAB_SIZE2)

# print("warp polar mode")
# print(cv2.WARP_POLAR_LINEAR)
# print(cv2.WARP_POLAR_LOG)

