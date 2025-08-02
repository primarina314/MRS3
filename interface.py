import numpy as np
import cv2
import time
import os
from collections import defaultdict
from PIL import Image
import configparser
import interpolation as inter
import gc
import utils
import mrs3 as mr
import uuid
from pathlib import Path
from ultralytics import YOLO
import zipfile
from dataclasses import dataclass

"""
압축
input: png 여러 개 -> 업로드한 png 파일들을 한 폴더 내부에 모아놓고 순차로 처리
output: pkg 여러 개 또는 pkg 담긴 zip

1. 업로드된 이미지들을 서버 내 한 폴더에 저장 - 폴더명은 현재시각을 이용하거나, 압축처리가 끝난 폴더는 삭제하여 폴더명이 겹치지 않게 설정
2. 폴더 내 이미지 순차로 pkg 변환 - 자동 타겟 설정(얼굴 인식) 옵션 추가. 자동 옵션 선택 시 모든 이미지 자동 처리. 수동 옵션 선택 시 모두 자동 처리
→ 업로드한 이미지 중 일부만 선택해서 자동 타겟 되도록 하고 나머지는 수동 타겟. 모두 선택하면 전체가 자동 처리
→ 자동처리 선택된 이미지들과 수동처리 이미지들을 각기 다른 폴더에 저장해서 구분
→ 수동타겟은 기존처럼 폴리곤 형태로 사용자가 직접 처리. 자동타겟은 얼굴(또는 사물)인식하여 자동처리.
3. 모든 pkg(수동, 자동 구분 없이 한 번에) 모아서 zip 생성(복원 시에는 자동 수동 구분 의미 없으므로)
4. client 에게 zip 리턴

(pkg 압축 후 리턴하기 전에 미리보기 옵션 추가 - 미리보기 후에 복원결과가 별로다 싶으면 zip 에서 제외하는 기능도 추가)

복원
input: pkg 여러 개 또는 pkg 담긴 zip
output: png 여러 개(이중에서 선택) 또는 png 담긴 zip

1. 업로드된 파일형식이 zip 이면 압축해제하여 한 폴더에 저장. pkg 여러개면 pkg 형식으로 한 폴더에 저장 → pkg 가 한 폴더 안에 모여있도록.
2. 폴더 내 pkg 파일들을 순차로 돌면서 기존 해상도 png 로 복원 → 또 다른 폴더 하나에 png 모아두기. png 이미지 파일명은 기존 pkg 와 같게 설정 후 뒤에 -restored 만 붙여서 설정.
3. 모든 처리가 완료되면 사용자에게 미리보기 기능 제공. 이 중 사용자가 일부(혹은 전체) 선택하여 (a) 개별 다운로드 또는 (b) 선택된 이미지들을 한 번에 다운로드(한 번에 png 여러 장 다운로드) 또는 (c) 선택 이미지들 모아서 zip 압축 리턴
"""


"""
수동: 기존대로 front 에서 불러온 좌표 리스트 대로 처리하는 작업
front 에서 폴더 내 모든 이미지의 좌표를 불러왔다고 가정 -> 리스트에 순차저장(이름순)


"""

TEMP_DIR = "temp"

def get_unique_path(filename: str, suffix: str = "") -> str:
    """uuid와 원본 파일명, 옵션 suffix로 유니크 경로 생성."""
    session_id = str(uuid.uuid4())
    safe_name = Path(filename).name  # 보안: 디렉토리 오염 방지
    return os.path.join(TEMP_DIR, f"{session_id}_{suffix}{safe_name}")

roi_contours = []
roi_contour_num = 0

# len(contours) == contour_num -> 새로 추가
# len(contours) > contour_num -> 마지막 요소(contour_num번쨰 index)에 그대로 추가

def _draw_multiple_polygon(event, x, y, flags, param):
    global roi_contours, roi_contour_num
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_contours) == roi_contour_num:
            roi_contours.append([])
        roi_contours[roi_contour_num].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and roi_contours and len(roi_contours)!=roi_contour_num and len(roi_contours[-1]) >= 3:
        roi_contour_num += 1


def _select_multiple_polygon_roi(image_path):

    """
    타겟 다중 선택
    우클릭으로 다음 타겟으로 넘어가고, s키 눌러서 최종 저장
    리턴: (타겟 원본 부분 ndarray, 타겟 바이너리 mask, (y, y+h, x, x+w)) 튜플
    """
    global drawing, roi_contours, roi_contour_num
    img = cv2.imread(image_path)
    clone = img.copy()
    cv2.namedWindow("indicate polygon in img")
    cv2.setMouseCallback("indicate polygon in img", _draw_multiple_polygon)

    drawing = True
    roi_contours = []
    roi_contour_num = 0
    while drawing:
        temp = clone.copy()

        for i in range(len(roi_contours)):
            points = roi_contours[i]
            if len(points) > 0:
                cv2.polylines(temp, [np.array(points)], i < roi_contour_num, (0,255,0), 2)
                for pt in points:
                    cv2.circle(temp, pt, 3, (0,0,255), -1)
        cv2.imshow("indicate polygon in img", temp)

        key = cv2.waitKey(1)
        if key == 27:  # ESC로 취소
            roi_contours = []
            drawing = False
            break
        if key == ord('s') and roi_contours and len(roi_contours[-1]) >= 3:  # 's'로 저장
            drawing = False

    cv2.destroyAllWindows()

    result = roi_contours.copy()
    roi_contours = []
    roi_contour_num = 0
    
    return result

model = YOLO('models/yolov8m-face-lindevs.pt')

def compress_mult_img_server(
        input_path: str, 
        output_path: str, 
        manual=True, 
        scaler=4, 
        interpolation=mr.INTER_AREA
        ):
    """
    :param input_path: 수동 압축 대상 이미지 모아놓은 폴더 경로
    :param output_path: pkg 결과 저장 폴더 경로
    :param manual: 해당 폴더 처리 자동/수동 여부. True 면 수동, False 면 자동.
    :param scaler: 이미지 shrink scaler
    :param interpolation: shrink 에 적용할 interpolation manner
    """

    target_path = output_path
    fn, ext = os.path.splitext(output_path)
    if ext.lower() == '.zip':
        target_path = get_unique_path(fn, suffix="pkgs-folder_")

    for filename in os.listdir(input_path):
        if filename.lower().endswith('.png'):
            full_path = os.path.join(input_path, filename)
            
            filename_with_ext = os.path.basename(filename)
            pkg_filename_split, _ = os.path.splitext(filename_with_ext)
            pkg_filename = f'{pkg_filename_split}.pkg'

            roi_point_lists = []
            # roi_point_lists 의 리스트 - 서로 매칭돼야함
            # iteration 의 filename 과 매칭되는 roi_point_lists 가 뭔지 알아야 함
            # 아니면 이미지와 같은 이름의 ini 파일에 적어둔다던가
            if manual: # 수동 타겟
                # TODO: 현재 full_path 이미지에 대응하는 roi_point_lists 가져오기 - 아래 코드 지우고, 웹형식에 맞게 수정 필요
                roi_point_lists = _select_multiple_polygon_roi(full_path)
            else: # 자동 타겟
                img = cv2.imread(full_path)
                results = model(img)
                
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    roi_point_lists.append([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
            
            mr.compress_img_mult_tgs_server(img_path=full_path, 
                                            output_path=target_path, 
                                            scaler=scaler, 
                                            pkg_filename=pkg_filename,
                                            roi_point_lists=roi_point_lists,
                                            interpolation=interpolation,
                                            delete_temp=True)
    if target_path != output_path: # output_path 가 zip 이면 zip 으로 압축해서 저장
        zip_folder_server(target_path, output_path)
    return

@dataclass
class CompressInputInfo:
    path: str
    manual: bool = True
    scaler: int = 4
    interpolation: int = mr.INTER_AREA


def compress_mult_imgs_in_mult_folders_server(
        input_infos: list[CompressInputInfo], 
        output_path: str
        ) -> None:
    """
    다수 폴더 내 이미지 한 번에 처리
    e.g. 자동 타겟 이미지 모아놓은 폴더 및 수동타겟 모은 폴더 한 번에 묶어서 처리
    """

    target_path = output_path
    fn, ext = os.path.splitext(output_path)
    if ext.lower() == '.zip':
        target_path = get_unique_path(fn, suffix="pkgs-folder_")
    
    for info in input_infos:
        compress_mult_img_server(input_path=info.path,
                                 output_path=target_path,
                                 manual=info.manual,
                                 scaler=info.scaler,
                                 interpolation=info.interpolation
                                 )
    
    if target_path != output_path: # output_path 가 zip 이면 zip 으로 압축해서 저장
        zip_folder_server(target_path, output_path)

def restore_imgs_in_folder_server(input_path: str, output_path: str, mrs3_mode):

    """
    input_path 확장자가 .zip 이면 자동으로 압축해제해서 처리
    """

    target_path = input_path

    fn, ext = os.path.splitext(input_path)
    if ext.lower() == '.zip':
        unzip_path = get_unique_path(fn, suffix="unzip_")
        unzip_server(input_path, unzip_path)
        target_path = unzip_path

    for filename in os.listdir(target_path):
        if filename.lower().endswith('.pkg'):
            full_path = os.path.join(target_path, filename)

            filename_with_ext = os.path.basename(filename)
            img_filename_split, _ = os.path.splitext(filename_with_ext)
            img_filename = f'{img_filename_split}.png'

            unpack_path = get_unique_path(img_filename_split, suffix="unpacked_")
            utils.unpack_files(full_path, unpack_path)

            mr.restore_img_mult_tgs_server(input_path=unpack_path, 
                                           mrs3_mode=mrs3_mode, 
                                           output_path=output_path, 
                                           img_filename=img_filename)

def zip_folder_server(folder_path: str, zip_filename: str):
    """
    :param folder_path: 폴더 경로
    :param zip_filename: 저장할 zip 파일명(e.g. compression.zip)
    """
    # 지정 폴더 내 모든 파일을 zip 으로 묶어서 저장
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # 파일인지 확인
                # 압축 내 파일 이름은 파일명만 사용 (전체 경로 아님)
                zipf.write(file_path, arcname=filename)
    return


def unzip_server(zip_path: str, extract_folder: str):
    """
    :param zip_path: 압축해제하고자 하는 zip 파일 경로(file.zip)
    :param extract_folder: 압축해제하여 파일들을 저장할 폴더 경로
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)  # 모든 파일을 지정한 폴더에 압축 해제
    return

