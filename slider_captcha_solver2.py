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
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class AdvancedSliderSolver:
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
        self.model_path = "slider_model.pkl"
        self.data_path = "slider_data.json"
        self.model = None
        self.feature_data = []
        self.label_data = []
        
    def extract_number(self, css_value):
        """从CSS值中提取数字，处理小数"""
        match = re.search(r'([\d.]+)', css_value)
        if match:
            return float(match.group(1))
        return 0
    
    def get_captcha_image(self):
        """获取验证码图片"""
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
            
            return bg_image
        
        except Exception as e:
            print(f"获取验证码图片时出错: {str(e)}")
            raise
    
    def extract_image_features(self, image, x_position):
        """从图像中提取特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 提取感兴趣区域 (ROI) - 假设滑块高度在中间部分
        height, width = gray.shape
        roi_height = 60
        roi_top = (height - roi_height) // 2
        roi = gray[roi_top:roi_top+roi_height, :]
        
        # 特征1: 水平方向的梯度变化
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_abs = np.abs(sobel_x)
        
        # 特征2: 垂直方向的梯度变化
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y_abs = np.abs(sobel_y)
        
        # 特征3: 边缘密度
        edges = cv2.Canny(roi, 50, 150)
        
        # 特征4: 局部二值模式 (LBP) - 纹理特征
        lbp = self.local_binary_pattern(roi, 8, 1)
        
        # 特征5: 在x_position附近的窗口特征
        window_size = 10
        x_start = max(0, x_position - window_size)
        x_end = min(width, x_position + window_size)
        
        # 提取窗口内的特征
        window_roi = roi[:, x_start:x_end]
        if window_roi.size > 0:
            window_mean = np.mean(window_roi)
            window_std = np.std(window_roi)
            window_edges = np.sum(edges[:, x_start:x_end]) / (window_roi.shape[0] * window_roi.shape[1])
        else:
            window_mean = 0
            window_std = 0
            window_edges = 0
        
        # 组合所有特征
        features = [
            np.mean(sobel_x_abs), np.std(sobel_x_abs),
            np.mean(sobel_y_abs), np.std(sobel_y_abs),
            np.sum(edges) / (roi.shape[0] * roi.shape[1]),
            np.mean(lbp), np.std(lbp),
            window_mean, window_std, window_edges,
            x_position / width  # 归一化位置
        ]
        
        return features
    
    def local_binary_pattern(self, image, points, radius):
        """计算局部二值模式 (LBP)"""
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0]-radius):
            for j in range(radius, image.shape[1]-radius):
                center = image[i, j]
                binary = 0
                for p in range(points):
                    # 计算采样点坐标
                    r = i + radius * np.cos(2 * np.pi * p / points)
                    c = j - radius * np.sin(2 * np.pi * p / points)
                    
                    # 双线性插值
                    r0, c0 = int(r), int(c)
                    r1, c1 = r0 + 1, c0 + 1
                    
                    # 处理边界情况
                    r0 = max(0, min(r0, image.shape[0]-1))
                    r1 = max(0, min(r1, image.shape[0]-1))
                    c0 = max(0, min(c0, image.shape[1]-1))
                    c1 = max(0, min(c1, image.shape[1]-1))
                    
                    # 双线性插值
                    fr = r - r0
                    fc = c - c0
                    w1 = (1 - fr) * (1 - fc)
                    w2 = fr * (1 - fc)
                    w3 = (1 - fr) * fc
                    w4 = fr * fc
                    
                    interpolated = (w1 * image[r0, c0] + w2 * image[r1, c0] + 
                                   w3 * image[r0, c1] + w4 * image[r1, c1])
                    
                    # 设置二进制位
                    if interpolated >= center:
                        binary |= (1 << p)
                
                lbp[i, j] = binary
        
        return lbp
    
    def find_target_by_comparison(self, image):
        """通过图像对比找到目标位置"""
        height, width = image.shape[:2]
        
        # 生成多个候选位置
        candidates = []
        for x in range(30, width - 30, 5):  # 从30到width-30，步长为5
            candidates.append(x)
        
        # 为每个候选位置提取特征
        candidate_features = []
        for x in candidates:
            features = self.extract_image_features(image, x)
            candidate_features.append((x, features))
        
        # 如果没有训练好的模型，使用启发式方法
        if self.model is None:
            # 使用简单的启发式：寻找梯度变化最大的位置
            best_x = self.find_by_gradient_heuristic(image)
            print(f"使用启发式方法找到位置: {best_x}")
            return best_x
        
        # 使用机器学习模型预测
        X = [features for _, features in candidate_features]
        predictions = self.model.predict_proba(X)
        
        # 选择概率最高的位置
        best_idx = np.argmax(predictions[:, 1])  # 假设第二列是"正确"的概率
        best_x = candidates[best_idx]
        confidence = predictions[best_idx, 1]
        
        print(f"使用机器学习模型找到位置: {best_x}, 置信度: {confidence:.2f}")
        
        return best_x
    
    def find_by_gradient_heuristic(self, image):
        """使用梯度启发式方法找到目标位置"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 计算水平方向的梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_abs = np.abs(sobel_x)
        
        # 计算每列的梯度总和
        column_gradients = np.sum(sobel_x_abs, axis=0)
        
        # 找到梯度变化最大的位置
        # 使用滑动窗口计算局部最大值
        window_size = 10
        max_gradient = 0
        best_x = gray.shape[1] // 2  # 默认中间位置
        
        for x in range(window_size, len(column_gradients) - window_size):
            # 计算窗口内的梯度总和
            window_sum = np.sum(column_gradients[x-window_size:x+window_size])
            
            if window_sum > max_gradient:
                max_gradient = window_sum
                best_x = x
        
        return best_x
    
    def load_model(self):
        """加载训练好的模型"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("已加载训练好的模型")
                return True
            except Exception as e:
                print(f"加载模型失败: {e}")
        
        return False
    
    def save_model(self):
        """保存训练好的模型"""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            print("模型已保存")
    
    def load_data(self):
        """加载训练数据"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.feature_data = data.get('features', [])
                    self.label_data = data.get('labels', [])
                print(f"已加载 {len(self.feature_data)} 条训练数据")
            except Exception as e:
                print(f"加载训练数据失败: {e}")
    
    def save_data(self):
        """保存训练数据"""
        data = {
            'features': self.feature_data,
            'labels': self.label_data
        }
        with open(self.data_path, 'w') as f:
            json.dump(data, f)
        print(f"训练数据已保存，共 {len(self.feature_data)} 条")
    
    def add_training_data(self, image, target_x, success):
        """添加训练数据"""
        features = self.extract_image_features(image, target_x)
        label = 1 if success else 0
        
        self.feature_data.append(features)
        self.label_data.append(label)
        
        # 限制数据量，避免过大
        if len(self.feature_data) > 1000:
            self.feature_data = self.feature_data[-1000:]
            self.label_data = self.label_data[-1000:]
    
    def train_model(self):
        """训练模型"""
        if len(self.feature_data) < 10:
            print("训练数据不足，无法训练模型")
            return False
        
        X = np.array(self.feature_data)
        y = np.array(self.label_data)
        
        # 检查是否有正负样本
        if np.sum(y) == 0 or np.sum(y) == len(y):
            print("训练数据缺乏多样性，无法训练模型")
            return False
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型训练完成，测试集准确率: {accuracy:.2f}")
        
        # 保存模型
        self.save_model()
        
        return True
    
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
    
    def solve_captcha(self, max_attempts=5):
        """破解验证码的主要函数"""
        # 加载模型和训练数据
        self.load_data()
        model_loaded = self.load_model()
        
        for attempt in range(max_attempts):
            try:
                print(f"尝试第 {attempt+1} 次破解验证码...")
                
                # 获取验证码图片
                bg_image = self.get_captcha_image()
                
                # 找到目标位置
                target_x = self.find_target_by_comparison(bg_image)
                
                print(f"推断的滑块目标位置: x={target_x}")
                
                # 获取滑块元素
                slider_element = self.driver.find_element(By.ID, "slider")
                
                # 拖动滑块
                self.drag_slider(slider_element, target_x)
                
                # 点击验证按钮
                if self.click_verify_button():
                    # 检查验证结果
                    result = self.check_result()
                    
                    # 记录训练数据
                    self.add_training_data(bg_image, target_x, result)
                    
                    if result:
                        # 验证成功，保存数据并尝试训练模型
                        self.save_data()
                        if len(self.feature_data) >= 20 and not model_loaded:
                            self.train_model()
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
        # 保存收集到的数据
        self.save_data()
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
        solver = AdvancedSliderSolver(driver)
        
        # 尝试破解验证码
        success = solver.solve_captcha(max_attempts=10)
        
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