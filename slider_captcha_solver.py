import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import random
import requests
from io import BytesIO
from PIL import Image
import re

class SliderCaptchaSolver:
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
    
    def extract_number(self, css_value):
        """从CSS值中提取数字，处理小数"""
        match = re.search(r'([\d.]+)', css_value)
        if match:
            return float(match.group(1))
        return 0
    
    def get_captcha_info(self):
        """获取验证码信息"""
        try:
            # 等待页面加载
            self.wait.until(EC.presence_of_element_located((By.ID, "bgImage")))
            
            # 获取背景图URL
            bg_img_url = self.driver.execute_script("return document.getElementById('bgImage').src")
            
            if not bg_img_url or bg_img_url == "":
                print("背景图URL为空，等待图片加载...")
                time.sleep(2)
                bg_img_url = self.driver.execute_script("return document.getElementById('bgImage').src")
            
            # 下载背景图
            bg_response = requests.get(bg_img_url)
            bg_image = Image.open(BytesIO(bg_response.content))
            bg_image = np.array(bg_image)
            
            # 获取洞口位置信息
            hole_element = self.driver.find_element(By.ID, "hole")
            hole_x = self.extract_number(hole_element.value_of_css_property("left"))
            hole_y = self.extract_number(hole_element.value_of_css_property("top"))
            
            print(f"检测到洞口位置: x={hole_x}, y={hole_y}")
            
            return bg_image, hole_x, hole_y
        
        except Exception as e:
            print(f"获取验证码信息时出错: {str(e)}")
            raise
    
    def detect_slider_position(self, bg_image, hole_x, hole_y):
        """检测滑块应该移动到的正确位置"""
        # 在你的验证码实现中，正确的目标位置就是洞口位置
        # 因为滑块需要正好覆盖洞口
        return hole_x
    
    def generate_track(self, distance):
        """生成滑动轨迹，模拟人类行为"""
        distance = abs(distance)
        
        # 如果距离太小，直接返回一个小的移动
        if distance < 5:
            return [int(distance)]
            
        # 初始速度
        v = 0
        # 初始时间
        t = 0.2
        # 轨迹列表
        tracks = []
        # 当前位移
        current = 0
        # 减速阈值
        mid = distance * 3 / 5
        
        while current < distance:
            if current < mid:
                # 加速度
                a = 2
            else:
                # 减速度
                a = -3
                
            # 初始速度
            v0 = v
            # 当前速度
            v = v0 + a * t
            # 移动距离
            move = v0 * t + 0.5 * a * t * t
            # 当前位移
            current += move
            # 加入轨迹
            tracks.append(round(move))
        
        # 微调，确保总位移正好等于distance
        while sum(tracks) > distance:
            tracks[-1] -= 1
        
        if sum(tracks) < distance:
            tracks.append(distance - sum(tracks))
            
        return tracks
    
    def drag_slider(self, slider_element, distance):
        """拖动滑块"""
        # 确保距离是正数
        abs_distance = abs(distance)
        
        # 生成轨迹
        tracks = self.generate_track(abs_distance)
        
        print(f"开始拖动滑块，距离: {abs_distance}, 轨迹: {tracks}")
        
        # 按下滑块
        ActionChains(self.driver).click_and_hold(slider_element).perform()
        time.sleep(0.2)
        
        # 按照轨迹移动
        for track in tracks:
            ActionChains(self.driver).move_by_offset(track, 0).perform()
            # 随机延迟，模拟人类行为
            time.sleep(random.uniform(0.01, 0.05))
        
        # 释放滑块
        ActionChains(self.driver).release().perform()
        time.sleep(0.5)
    
    def click_verify_button(self):
        """点击验证按钮"""
        try:
            verify_btn = self.driver.find_element(By.ID, "verifyBtn")
            verify_btn.click()
            print("已点击验证按钮")
            return True
        except Exception as e:
            print(f"点击验证按钮时出错: {str(e)}")
            return False
    
    def check_result(self):
        """检查验证结果"""
        try:
            # 等待可能的弹窗
            time.sleep(1)
            
            # 检查是否有alert弹窗
            alert = self.driver.switch_to.alert
            alert_text = alert.text
            alert.accept()
            
            if "Successful" in alert_text:
                print("验证成功!")
                return True
            else:
                print(f"验证失败: {alert_text}")
                return False
        except:
            # 没有alert，检查页面是否有其他成功提示
            try:
                # 这里可以根据你的页面实际成功提示来调整
                success_indicator = self.driver.find_element(By.CLASS_NAME, "success-message")
                if success_indicator.is_displayed():
                    print("验证成功!")
                    return True
            except:
                pass
            
            print("验证结果未知")
            return False
    
    def solve_captcha(self, max_attempts=3):
        """破解验证码的主要函数"""
        for attempt in range(max_attempts):
            try:
                print(f"尝试第 {attempt+1} 次破解验证码...")
                
                # 获取验证码信息
                bg_image, hole_x, hole_y = self.get_captcha_info()
                
                # 检测滑块应该移动到的位置
                target_x = self.detect_slider_position(bg_image, hole_x, hole_y)
                
                # 计算需要滑动的距离
                distance = target_x
                
                print(f"需要滑动的距离: {distance}")
                
                # 获取滑块元素
                slider_element = self.driver.find_element(By.ID, "slider")
                
                # 拖动滑块
                self.drag_slider(slider_element, distance)
                
                # 点击验证按钮
                if self.click_verify_button():
                    # 检查验证结果
                    result = self.check_result()
                    
                    if result:
                        return True
                    else:
                        print("验证失败，重试中...")
                        # 点击刷新按钮
                        refresh_btn = self.driver.find_element(By.ID, "refreshBtn")
                        refresh_btn.click()
                        time.sleep(1)
                        continue  # 继续下一次尝试
                else:
                    print("无法点击验证按钮，重试中...")
                    # 点击刷新按钮
                    refresh_btn = self.driver.find_element(By.ID, "refreshBtn")
                    refresh_btn.click()
                    time.sleep(1)
                    
            except Exception as e:
                print(f"验证过程中出现错误: {str(e)}")
                # 点击刷新按钮重试
                try:
                    refresh_btn = self.driver.find_element(By.ID, "refreshBtn")
                    refresh_btn.click()
                    time.sleep(1)
                except:
                    pass
        
        print(f"经过 {max_attempts} 次尝试后验证失败")
        return False


def main():
    # 设置Chrome浏览器选项
    options = webdriver.ChromeOptions()
    # 取消显示"Chrome正在受到自动软件控制"的提示
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # 初始化浏览器驱动
    driver = webdriver.Chrome(options=options)
    
    try:
        # 打开验证码页面
        print("正在打开验证码页面...")
        driver.get("http://localhost:5000/slider")
        
        # 等待页面加载
        time.sleep(2)
        
        # 初始化破解器
        solver = SliderCaptchaSolver(driver)
        
        # 尝试破解验证码
        success = solver.solve_captcha(max_attempts=5)
        
        if success:
            print("验证码破解成功!")
        else:
            print("验证码破解失败!")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    
    finally:
        # 等待一段时间后关闭浏览器
        input("按回车键关闭浏览器...")
        driver.quit()


if __name__ == "__main__":
    main()