import cv2
import numpy as np
import time
import os
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class CatLitterCaptchaSolver:
    def __init__(self, headless=False):
        self.driver = None
        self.canvas = None
        self.frames = []
        self.headless = headless
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_browser(self):
        """启动浏览器"""
        self.logger.info("Starting browser...")
        
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1200,800')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.get("http://localhost:5000/cat")
            
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "catLitterCanvas"))
            )
            
            self.canvas = self.driver.find_element(By.ID, "catLitterCanvas")
            self.logger.info("Browser started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Browser startup failed: {e}")
            return False
    
    def capture_canvas_frame(self, filename=None):
        """捕获canvas的当前帧"""
        try:
            if filename is None:
                filename = f"temp_frame_{int(time.time())}.png"
                
            self.canvas.screenshot(filename)
            frame = cv2.imread(filename)
            
            # 清理临时文件
            if os.path.exists(filename):
                os.remove(filename)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None
    
    def simulate_mouse_movements(self, pattern_type="comprehensive"):
        """模拟鼠标移动来激活流动性效果"""
        self.logger.info("Simulating mouse movements...")
        
        actions = ActionChains(self.driver)
        canvas_rect = self.canvas.rect
        width = canvas_rect['width']
        height = canvas_rect['height']
        
        if pattern_type == "comprehensive":
            # 组合多种移动模式
            self._simulate_circular_movement(actions, width, height)
            self._simulate_random_movement(actions, width, height, 8)
            self._simulate_grid_movement(actions, width, height)
            self._simulate_sweeping_movement(actions, width, height)
            
        elif pattern_type == "circular":
            self._simulate_circular_movement(actions, width, height)
            
        elif pattern_type == "random":
            self._simulate_random_movement(actions, width, height, 15)
            
        self.logger.info("Mouse movements completed")
    
    def _simulate_circular_movement(self, actions, width, height):
        """模拟圆形移动"""
        center_x = width // 2
        center_y = height // 2
        radius = min(center_x, center_y) - 30
        
        for angle in range(0, 360, 15):
            rad = np.radians(angle)
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            
            actions.move_to_element_with_offset(self.canvas, int(x), int(y))
            actions.perform()
            time.sleep(0.03)
    
    def _simulate_random_movement(self, actions, width, height, num_points):
        """模拟随机移动"""
        for i in range(num_points):
            x = np.random.randint(30, width - 30)
            y = np.random.randint(30, height - 30)
            
            actions.move_to_element_with_offset(self.canvas, x, y)
            actions.perform()
            time.sleep(0.1)
    
    def _simulate_grid_movement(self, actions, width, height):
        """模拟网格移动"""
        step_x = width // 6
        step_y = height // 4
        
        for y in range(step_y, height, step_y):
            for x in range(step_x, width, step_x):
                actions.move_to_element_with_offset(self.canvas, x, y)
                actions.perform()
                time.sleep(0.05)
    
    def _simulate_sweeping_movement(self, actions, width, height):
        """模拟扫动移动"""
        # 水平扫动
        for x in range(50, width - 50, 10):
            y = height // 2
            actions.move_to_element_with_offset(self.canvas, x, y)
            actions.perform()
            time.sleep(0.02)
        
        # 垂直扫动
        for y in range(50, height - 50, 10):
            x = width // 2
            actions.move_to_element_with_offset(self.canvas, x, y)
            actions.perform()
            time.sleep(0.02)
    
    def capture_motion_sequence(self, num_sequences=3):
        """捕获运动序列"""
        self.logger.info("Starting motion sequence capture...")
        self.frames = []
        
        for i in range(num_sequences):
            self.logger.info(f"Sequence {i+1}/{num_sequences}")
            
            # 捕获移动前的帧
            frame_before = self.capture_canvas_frame()
            if frame_before is not None:
                self.frames.append(("before", frame_before))
            
            # 模拟移动
            movement_patterns = ["circular", "random", "sweeping"]
            pattern = movement_patterns[i % len(movement_patterns)]
            self.simulate_mouse_movements(pattern)
            
            # 捕获移动后的帧
            frame_after = self.capture_canvas_frame()
            if frame_after is not None:
                self.frames.append(("after", frame_after))
            
            time.sleep(0.5)
        
        self.logger.info(f"Captured {len(self.frames)} frames total")
        return len(self.frames) > 0
    
    def detect_static_regions(self):
        """检测静态区域（猫屎）"""
        if len(self.frames) < 4:
            self.logger.error("Insufficient frames for analysis")
            return []
        
        self.logger.info("Detecting static regions...")
        
        # 提取所有帧
        frames_data = [frame[1] for frame in self.frames]
        
        # 转换为灰度图
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_data]
        
        # 计算帧间差异矩阵
        diff_matrix = np.zeros(gray_frames[0].shape, dtype=np.float32)
        
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            diff_matrix += diff.astype(np.float32)
        
        # 归一化差异矩阵
        if len(gray_frames) > 1:
            diff_matrix /= (len(gray_frames) - 1)
        
        # 找到静态区域（差异很小的区域）
        static_threshold = 8  # 可调整的阈值
        static_mask = diff_matrix < static_threshold
        
        # 应用形态学操作清理噪声
        kernel = np.ones((5, 5), np.uint8)
        static_mask = cv2.morphologyEx(static_mask.astype(np.uint8), 
                                     cv2.MORPH_OPEN, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)
        
        # 找到轮廓
        contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        detected_clumps = []
        min_area = 150  # 最小区域面积
        max_area = 2000  # 最大区域面积
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_clumps.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'center_x': center_x, 'center_y': center_y,
                    'area': area
                })
        
        # 按面积排序，取前几个最大的区域
        detected_clumps.sort(key=lambda x: x['area'], reverse=True)
        max_clumps = 5  # 最多检测5个块状物
        detected_clumps = detected_clumps[:max_clumps]
        
        self.logger.info(f"Detected {len(detected_clumps)} static regions")
        return detected_clumps
    
    def refine_detection_with_multiple_approaches(self):
        """使用多种方法精炼检测结果"""
        self.logger.info("Refining detection with multiple approaches...")
        
        all_detections = []
        
        # 方法1: 基于帧间差异
        diff_based = self.detect_static_regions()
        all_detections.extend(diff_based)
        
        # 方法2: 基于纹理分析
        texture_based = self.texture_based_detection()
        all_detections.extend(texture_based)
        
        # 合并和去重检测结果
        merged_detections = self.merge_detections(all_detections)
        
        self.logger.info(f"Refined detection: {len(merged_detections)} regions found")
        return merged_detections
    
    def texture_based_detection(self):
        """基于纹理分析的检测方法"""
        if not self.frames:
            return []
        
        # 使用最后一帧进行分析
        frame = self.frames[-1][1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算局部二值模式(LBP)纹理
        lbp = self.compute_lbp(gray)
        
        # 找到纹理均匀的区域（可能是块状物）
        uniform_regions = self.find_uniform_texture_regions(lbp)
        
        return uniform_regions
    
    def compute_lbp(self, image):
        """计算LBP纹理"""
        height, width = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] > center) << 7
                code |= (image[i-1, j] > center) << 6
                code |= (image[i-1, j+1] > center) << 5
                code |= (image[i, j+1] > center) << 4
                code |= (image[i+1, j+1] > center) << 3
                code |= (image[i+1, j] > center) << 2
                code |= (image[i+1, j-1] > center) << 1
                code |= (image[i, j-1] > center) << 0
                lbp[i, j] = code
                
        return lbp
    
    def find_uniform_texture_regions(self, lbp):
        """找到纹理均匀的区域"""
        # 计算每个像素的均匀性（跳变次数）
        uniform_map = np.zeros_like(lbp, dtype=np.uint8)
        
        for i in range(1, lbp.shape[0]-1):
            for j in range(1, lbp.shape[1]-1):
                # 计算二进制跳变次数
                binary = format(lbp[i, j], '08b')
                transitions = 0
                for k in range(8):
                    if binary[k] != binary[(k+1)%8]:
                        transitions += 1
                
                # 如果跳变次数少，认为是均匀纹理
                if transitions <= 2:
                    uniform_map[i, j] = 255
        
        # 找到轮廓
        contours, _ = cv2.findContours(uniform_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1500:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'center_x': x + w//2, 'center_y': y + h//2,
                    'area': area
                })
        
        return regions
    
    def merge_detections(self, detections):
        """合并重复的检测结果"""
        if not detections:
            return []
        
        # 简单的IOU合并
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            # 找到与当前检测重叠的其他检测
            group = [det1]
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                if self.calculate_iou(det1, det2) > 0.3:  # IOU阈值
                    group.append(det2)
                    used.add(j)
            
            # 合并组内的检测
            if len(group) > 1:
                merged_det = self.merge_detection_group(group)
                merged.append(merged_det)
            else:
                merged.append(det1)
            
            used.add(i)
        
        return merged
    
    def calculate_iou(self, det1, det2):
        """计算两个检测框的IOU"""
        x1 = max(det1['x'], det2['x'])
        y1 = max(det1['y'], det2['y'])
        x2 = min(det1['x'] + det1['width'], det2['x'] + det2['width'])
        y2 = min(det1['y'] + det1['height'], det2['y'] + det2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = det1['width'] * det1['height']
        area2 = det2['width'] * det2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def merge_detection_group(self, group):
        """合并一组检测结果"""
        if not group:
            return None
        
        # 计算平均位置和大小
        x = np.mean([det['x'] for det in group])
        y = np.mean([det['y'] for det in group])
        w = np.mean([det['width'] for det in group])
        h = np.mean([det['height'] for det in group])
        
        return {
            'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
            'center_x': int(x + w//2), 'center_y': int(y + h//2),
            'area': int(w * h)
        }
    
    def visualize_detection(self, clump_regions, filename="detection_result.png"):
        """可视化检测结果"""
        if not self.frames:
            self.logger.warning("No frame data for visualization")
            return
        
        # 使用最后一帧作为基础
        display_frame = self.frames[-1][1].copy()
        
        for i, clump in enumerate(clump_regions):
            x, y, w, h = clump['x'], clump['y'], clump['width'], clump['height']
            
            # 绘制矩形框
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(display_frame, (clump['center_x'], clump['center_y']), 
                      5, (0, 0, 255), -1)
            
            # 添加标签
            cv2.putText(display_frame, f"Clump {i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 保存结果
        cv2.imwrite(filename, display_frame)
        self.logger.info(f"Detection results saved to {filename}")
        
        # 显示结果（如果不在无头模式）
        if not self.headless:
            cv2.imshow("Detection Result", display_frame)
            cv2.waitKey(3000)  # 显示3秒
            cv2.destroyAllWindows()
    
    def drag_clumps_to_trash(self, clump_regions):
        """将检测到的块状物拖动到垃圾桶"""
        if not clump_regions:
            self.logger.warning("No clumps detected")
            return 0
        
        try:
            trash_bin = self.driver.find_element(By.CLASS_NAME, "trash-bin")
        except:
            self.logger.error("Trash bin element not found")
            return 0
        
        self.logger.info(f"Starting to drag {len(clump_regions)} clumps to trash bin...")
        
        success_count = 0
        for i, clump in enumerate(clump_regions):
            try:
                self.logger.info(f"Dragging clump {i+1}...")
                
                # 创建动作链
                actions = ActionChains(self.driver)
                
                # 移动到块状物中心并点击
                actions.move_to_element_with_offset(
                    self.canvas, clump['center_x'], clump['center_y']
                )
                actions.click_and_hold()
                
                # 移动到垃圾桶
                actions.move_to_element(trash_bin)
                actions.release()
                
                # 执行动作
                actions.perform()
                
                # 等待动画完成
                time.sleep(1.5)
                
                success_count += 1
                self.logger.info(f"Clump {i+1} dragged successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to drag clump {i+1}: {e}")
        
        self.logger.info(f"Successfully dragged {success_count}/{len(clump_regions)} clumps")
        return success_count
    
    def verify_solution(self):
        """验证是否破解成功"""
        try:
            # 检查是否跳转到了下一个验证码页面
            current_url = self.driver.current_url
            if "audio" in current_url:
                self.logger.info("Verification successful! Redirected to audio CAPTCHA page")
                return True
            
            # 检查成功消息
            message_element = self.driver.find_element(By.ID, "message")
            message_text = message_element.text
            if "成功" in message_text or "Success" in message_text:
                self.logger.info("Verification successful!")
                return True
                
            return False
            
        except:
            return False
    
    def solve_captcha(self, max_attempts=2):
        """完整的验证码破解流程"""
        self.logger.info("Starting cat litter CAPTCHA solving...")
        
        for attempt in range(max_attempts):
            self.logger.info(f"Attempt {attempt+1}/{max_attempts}")
            
            try:
                # 1. 启动浏览器
                if not self.setup_browser():
                    continue
                
                # 2. 捕获运动序列
                if not self.capture_motion_sequence():
                    self.logger.error("Motion sequence capture failed")
                    continue
                
                # 3. 检测静态区域
                clump_regions = self.refine_detection_with_multiple_approaches()
                
                if not clump_regions:
                    self.logger.warning("No clumps detected, trying alternative methods...")
                    # 备用方法：使用不同的移动模式
                    self.simulate_mouse_movements("random")
                    self.capture_motion_sequence(2)
                    clump_regions = self.detect_static_regions()
                
                if not clump_regions:
                    self.logger.error("All detection methods failed")
                    continue
                
                # 4. 可视化检测结果
                self.visualize_detection(clump_regions, f"attempt_{attempt+1}_result.png")
                
                # 5. 自动拖动到垃圾桶
                success_count = self.drag_clumps_to_trash(clump_regions)
                
                # 6. 验证结果
                time.sleep(2)  # 等待页面更新
                if self.verify_solution():
                    self.logger.info("🎉 CAPTCHA solved successfully!")
                    self.driver.quit()
                    return True
                elif success_count >= 2:  # 假设至少需要移除2个块状物
                    self.logger.info("Possible success but verification failed")
                else:
                    self.logger.warning("Solving failed, insufficient clumps removed")
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1} failed: {e}")
            
            finally:
                # 关闭浏览器准备下一次尝试
                if self.driver:
                    self.driver.quit()
                    self.driver = None
        
        self.logger.error("All attempts failed")
        return False
    
    def close(self):
        """清理资源"""
        if self.driver:
            self.driver.quit()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    print("=== Cat Litter CAPTCHA Solver ===")
    print("Make sure Flask app is running: python combined_app.py")
    print()
    
    # 选择模式
    print("Select mode:")
    print("1. Auto solve (headless mode)")
    print("2. Auto solve (with browser visible)")
    print("3. Detection only (no execution)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    solver = None
    try:
        if choice == "1":
            solver = CatLitterCaptchaSolver(headless=True)
            success = solver.solve_captcha()
            if success:
                print(" CAPTCHA solved successfully!")
            else:
                print(" CAPTCHA solving failed")
                
        elif choice == "2":
            solver = CatLitterCaptchaSolver(headless=False)
            success = solver.solve_captcha()
            if success:
                print("CAPTCHA solved successfully!")
            else:
                print(" CAPTCHA solving failed")
                
        elif choice == "3":
            solver = CatLitterCaptchaSolver(headless=False)
            solver.setup_browser()
            solver.capture_motion_sequence()
            clump_regions = solver.refine_detection_with_multiple_approaches()
            solver.visualize_detection(clump_regions)
            print("Detection completed, results saved")
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if solver:
            solver.close()


if __name__ == "__main__":
    main()