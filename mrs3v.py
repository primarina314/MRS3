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



