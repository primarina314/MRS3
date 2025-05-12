import tensorflow as tf
import numpy as np
import cv2
import time
import os
from enum import Enum
import keras_cv
import keras
import matplotlib.pyplot as plt
from multipledispatch import dispatch
import mrs3 as mr
import interpolation as inter


import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

@pipeline_def
def video_super_res_pipe(filenames, sequence_length, skip_vframes, step):
    # 비디오 리더 설정
    videos = fn.readers.video(
        device="gpu",
        filenames=filenames,
        sequence_length=sequence_length,
        skip_frames=skip_vframes,
        step=step
    )
    
    # 저해상도 변환 (다운샘플링)
    lr_frames = fn.resize(videos, resize_x=256, resize_y=256)
    
    return lr_frames, videos  # LR 및 HR 프레임 반환


# TODO: torch 설치(버전 확인 후 설치)

# TODO: MIA-VSR, EDVR 등 pretrained 모델 가져와서 테스트
# TODO: DEEPSORT, YOLO 활용 추적
# TODO: ESPCN, FSRCNN, LapSRN - 한 프레임(이미지) 적용 모델 알아보고 비교해보기 - 또는 비교하여 선택할 수 있도록 옵션

# TODO: 영상 파일 확장자
"""
영상 무손실 확장자는 AVI, TS, BMP, SVG 등이 있습니다. 특히 AVI는 무손실 압축을 지원하고, TS는 디지털 방송에서 원본 그대로 녹화할 때 사용되어 고화질 영상을 보장합니다
"""

