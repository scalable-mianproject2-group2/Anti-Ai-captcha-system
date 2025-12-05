# ---------------------------------------
# SECTION: slider captcha attacker3
# CONTRIBUTOR: ziwei zhao
# DESCRIPTION: slider captcha attacker
# ---------------------------------------
import time
import random
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class RealSliderSolver:
    def __init__(self):
        self.driver = None
        self.captcha_url = "http://127.0.0.1:5000/slider"
        
    def setup_browser(self):
        """设置浏览器"""
        print("正在启动浏览器...")
        chrome_options = Options()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def open_captcha_page(self):
        """打开验证码页面"""
        print("正在打开验证码页面...")
        self.driver.get(self.captcha_url)
        self.driver.maximize_window()
        time.sleep(3)
        
        # 点击触发验证码
        try:
            trigger = self.driver.find_element(By.CSS_SELECTOR, ".geetest_radar_tip")
            trigger.click()
            print("已触发验证码显示")
            time.sleep(2)
        except:
            print("无法找到验证码触发按钮")
        
    def get_captcha_elements(self):
        """获取验证码相关元素"""
        try:
            # 获取滑块背景图
            bg_element = self.driver.find_element(By.CSS_SELECTOR, ".geetest_canvas_bg")
            # 获取缺口背景图  
            slice_element = self.driver.find_element(By.CSS_SELECTOR, ".geetest_canvas_slice")
            # 获取滑块按钮
            slider = self.driver.find_element(By.CSS_SELECTOR, ".geetest_slider_button")
            
            return bg_element, slice_element, slider
        except Exception as e:
            print(f"找不到验证码元素: {e}")
            return None, None, None
            
    def get_element_screenshot(self, element, filename):
        """截取特定元素的截图"""
        location = element.location
        size = element.size
        
        # 截取整个页面
        self.driver.save_screenshot('full_page.png')
        
        # 计算元素区域
        left = location['x']
        top = location['y']
        right = left + size['width']
        bottom = top + size['height']
        
        # 截取元素区域
        screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
        screenshot.save(filename)
        return np.array(screenshot)
    
    def detect_gap_position(self, bg_img, slice_img):
        """检测缺口位置 - 使用模板匹配"""
        # 转换为灰度图
        bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 模板匹配
        result = cv2.matchTemplate(bg_gray, slice_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"模板匹配置信度: {max_val}")
        
        if max_val > 0.5:  # 匹配阈值
            gap_x = max_loc[0]
            print(f"检测到缺口位置: x={gap_x}")
            return gap_x
        else:
            print("模板匹配失败，尝试边缘检测方法...")
            return self.detect_gap_by_edges(bg_img, slice_img)
    
    def detect_gap_by_edges(self, bg_img, slice_img):
        """通过边缘检测找到缺口位置"""
        # 边缘检测
        bg_edges = cv2.Canny(bg_img, 100, 200)
        slice_edges = cv2.Canny(slice_img, 100, 200)
        
        # 查找轮廓
        bg_contours, _ = cv2.findContours(bg_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        slice_contours, _ = cv2.findContours(slice_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的轮廓（可能是缺口）
        if bg_contours:
            largest_contour = max(bg_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 验证这个轮廓是否合理
            if 40 < w < 100 and 30 < h < 80:  # 缺口的典型尺寸
                print(f"通过边缘检测找到缺口: x={x}, w={w}, h={h}")
                return x
        
        print("边缘检测也失败")
        return 100  # 默认值，但很可能失败
    
    def human_like_drag(self, slider, start_x, start_y, end_x, end_y):
        """模拟人类拖动行为"""
        print(f"开始拖动: 从 ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
        
        # 移动到滑块
        pyautogui.moveTo(start_x, start_y, duration=random.uniform(0.5, 1.0))
        time.sleep(random.uniform(0.1, 0.3))
        
        # 按下鼠标
        pyautogui.mouseDown()
        time.sleep(random.uniform(0.1, 0.2))
        
        # 模拟人类拖动轨迹
        total_distance = end_x - start_x
        steps = random.randint(30, 50)
        
        current_x = start_x
        for i in range(steps):
            # 非线性加速和减速
            progress = i / steps
            if progress < 0.3:  # 开始阶段慢
                speed_factor = 0.3 + progress * 2
            elif progress > 0.7:  # 结束阶段慢
                speed_factor = 1.0 - (progress - 0.7) * 2
            else:  # 中间阶段快
                speed_factor = 1.0
            
            # 添加随机扰动
            step_distance = (total_distance / steps) * speed_factor
            current_x += step_distance + random.uniform(-2, 2)
            
            # 轻微的垂直移动
            current_y = start_y + random.randint(-3, 3)
            
            pyautogui.moveTo(current_x, current_y, duration=0.05)
            
            # 随机暂停
            if random.random() < 0.1:
                time.sleep(random.uniform(0.05, 0.1))
        
        # 可能的轻微过冲和修正
        if random.random() < 0.7:  # 70%的几率会过冲
            overshoot = random.randint(2, 8)
            pyautogui.moveTo(current_x + overshoot, start_y, duration=0.1)
            time.sleep(random.uniform(0.05, 0.15))
            pyautogui.moveTo(end_x, start_y, duration=0.1)
        
        # 释放鼠标
        time.sleep(random.uniform(0.1, 0.3))
        pyautogui.mouseUp()
        
        print("拖动完成")
    
    def solve_captcha(self):
        """解决验证码的主逻辑"""
        bg_element, slice_element, slider = self.get_captcha_elements()
        
        if not all([bg_element, slice_element, slider]):
            print("无法获取验证码元素")
            return False
        
        # 获取元素位置
        slider_location = slider.location
        slider_size = slider.size
        
        # 截取验证码图片
        bg_img = self.get_element_screenshot(bg_element, "background.png")
        slice_img = self.get_element_screenshot(slice_element, "slice.png")
        
        # 检测缺口位置
        gap_x = self.detect_gap_position(bg_img, slice_img)
        
        # 计算拖动距离
        bg_location = bg_element.location
        drag_distance = gap_x - 10  # 减去一些偏移量
        
        # 计算起始和结束位置
        start_x = slider_location['x'] + slider_size['width'] // 2
        start_y = slider_location['y'] + slider_size['height'] // 2
        end_x = start_x + drag_distance
        
        print(f"计算出的拖动距离: {drag_distance}px")
        
        # 执行拖动
        self.human_like_drag(slider, start_x, start_y, end_x, start_y)
        
        # 等待结果
        time.sleep(3)
        
        # 检查是否成功
        try:
            success_element = self.driver.find_element(By.CSS_SELECTOR, ".geetest_success_radar_tip")
            if success_element.is_displayed():
                print("验证码破解成功!")
                return True
        except:
            print("验证码破解失败")
            return False
    
    def run(self):
        """运行主流程"""
        try:
            self.setup_browser()
            self.open_captcha_page()
            
            # 尝试破解
            success = self.solve_captcha()
            
            if success:
                print(" 任务完成!")
            else:
                print(" 破解失败，可能需要调整算法")
                
            # 保持浏览器打开以便查看结果
            input("按Enter键关闭浏览器...")
                
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            if self.driver:
                self.driver.quit()

if __name__ == "__main__":
    print("开始真正的滑块验证码破解...")
    solver = RealSliderSolver()
    solver.run()