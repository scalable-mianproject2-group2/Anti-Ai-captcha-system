import time
import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io

# -------------------------------------------
# Extract angle of arrow from screenshot
# -------------------------------------------
def detect_arrow_angle(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise Exception("No contours found in arrow image!")

    # Take the largest contour (the arrow)
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.reshape(-1, 2)

    # PCA to compute direction vector
    mean = np.mean(cnt, axis=0)
    cov = np.cov((cnt - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Largest eigenvector → direction of arrow
    largest_vec = eigvecs[:, np.argmax(eigvals)]

    # Compute angle
    angle = np.degrees(np.arctan2(largest_vec[1], largest_vec[0]))

    # Convert to "CSS-style" (0° = UP arrow)
    angle = (angle + 90) % 360

    return angle

# -------------------------------------------
# Main bot logic
# -------------------------------------------
def solve_captcha():
    print("🤖 Attacker 2 — Vision PCA Bot Starting...")

    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)  # keeps browser open

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("http://127.0.0.1:5000/")

    time.sleep(2)

    # Get target arrow element
    target_el = driver.find_element(By.ID, "target-arrow")

    # Screenshot arrow as PNG
    png_bytes = target_el.screenshot_as_png
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img_np = np.array(img)

    # Detect rotation
    detected_angle = detect_arrow_angle(img_np)
    print(f"🎯 Detected target angle: {detected_angle:.2f}°")

    # Rotate the user arrow
    user_el = driver.find_element(By.ID, "user-arrow")
    actions = ActionChains(driver)

    # Click user arrow
    actions.move_to_element(user_el).click_and_hold().perform()
    time.sleep(0.3)

    # Drag around a circular motion
    for i in range(40):
        # Simulate rotation
        actions.move_by_offset(3, 0)
    actions.release().perform()

    # Inject JS to manually rotate arrow precisely
    script = f"""
    document.getElementById('user-arrow').style.transform =
        'rotate({detected_angle}deg)';
    """
    driver.execute_script(script)

    print("✔ Bot rotated arrow using PCA vision.")

    time.sleep(1)

    # Submit
    submit_btn = driver.find_element(By.TAG_NAME, "button")
    submit_btn.click()

    time.sleep(2)

    # Read result
    status = driver.find_element(By.ID, "status").text
    print(f"🟢 CAPTCHA RESULT → {status}")

    print("Browser will remain open for inspection.")

if __name__ == "__main__":
    solve_captcha()
