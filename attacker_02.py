from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time, math

URL = "http://127.0.0.1:5000/"

def get_rotation(style):
    """Extract angle from transform: rotate(Xdeg) string"""
    import re
    match = re.search(r'rotate\(([-\d.]+)deg\)', style)
    return float(match.group(1)) if match else 0.0


def solve_captcha():
    print("🔍 Attacker 2 — Dynamic Visual Bot Starting...")

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-infobars")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)
    driver.get(URL)

    time.sleep(1.5)   # Allow UI animations

    # ---- Locate HTML elements ----
    target = driver.find_element(By.ID, "target-arrow")
    user = driver.find_element(By.ID, "user-arrow")
    submit = driver.find_element(By.TAG_NAME, "button")

    print("✔ DOM elements located.")

    # ---- Detect dynamic positions ----
    target_box = target.find_element(By.XPATH, "..")
    user_box = user.find_element(By.XPATH, "..")

    # Continuing to watch the UI (box moving) before clicking
    for i in range(10):  
        target_style = target.get_attribute("style")
        angle = get_rotation(target_style)

        user_style = user.get_attribute("style")
        user_angle = get_rotation(user_style)

        print(f"  → Target angle: {angle}, User angle: {user_angle}")

        time.sleep(0.2)

    # ---- Move the user arrow ----
    actions = ActionChains(driver)

    target_final_angle = angle
    print(f"🎯 Final Target Angle Locked: {target_final_angle}")

    # Drag horizontally proportional to required angle change
    delta = target_final_angle - user_angle
    drag_pixels = int(delta * 2)  # Tune sensitivity

    source_element = user_box

    print(f"🛠 Dragging user arrow by {drag_pixels} px to match target.")

    actions.click_and_hold(source_element).move_by_offset(drag_pixels, 0).pause(0.5).release().perform()

    time.sleep(1)

    # ---- Submit ----
    submit.click()

    time.sleep(1)

    # ---- Fetch result ----
    result = driver.find_element(By.ID, "status").text
    print("🟢 CAPTCHA RESULT →", result)

    time.sleep(3)
    driver.quit()


if __name__ == "__main__":
    solve_captcha()
