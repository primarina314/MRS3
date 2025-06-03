# 각종 유틸, 보조함수 등

"""
1. 웹 연결 후 압축 추가로 적용
2. 새로운 파일 확장자
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import os
import time

def compress_imgpresso(img_path, output_path):
    """
    img_path: absolute path of img
    output_path: absolute path of output folder
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new") # 헤드리스 모드

    prefs = {
        "download.default_directory": output_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 30)  # 대기 시간 30초

    try:
        # 1. 홈페이지 접속
        driver.get("https://imgpresso.co.kr")
        
        # 2. 이미지 업로드
        upload_input = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//input[@type='file']")))
        upload_input.send_keys(img_path)
        
        # 3. 옵션 선택(압축비율)
        processing_complete = wait.until(EC.visibility_of_element_located(
            (By.XPATH, "//div[contains(@id, 'conv-area-upload')]")))

        option_xpath = "//div[@id='b-ip-profile-2']"
        option_element = wait.until(EC.element_to_be_clickable(
            (By.XPATH, option_xpath)))
        option_element.click()
        
        # 4. 다운로드 버튼 클릭
        confirm_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@id='b-ip-upload-ok']")))        
        confirm_btn.click()
        
        # 5. 처리 완료 대기
        processing_complete = wait.until(EC.visibility_of_element_located(
            (By.XPATH, "//div[contains(@id, 'conv-area-complete')]")))
        
        # 6. 다운로드 버튼 클릭
        download_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[contains(@id, 'b-ip-download')]")))
        download_btn.click()
        
        # 7. 다운로드 완료 확인
        # TODO: 아래 코드에선 이미 다운로드된 png 파일에 의해 정상적으로 작동 안 됨 -> 새로운 png 만 인식하도록 수정할 필요 있음.
        start_time = time.time()
        while not any(fname.endswith('.png') for fname in os.listdir(output_path)):
            if time.time() - start_time > 60:
                raise TimeoutError("다운로드 시간 초과")
            time.sleep(1)
        
    finally:
        time.sleep(1) # 보조장치 -> 10초는 너무 길고, 줄일 필요 있음
        driver.quit()



    return


