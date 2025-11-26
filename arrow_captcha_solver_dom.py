from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

def solve_captcha():
    print("Attacker 1 (Python DOM Bot) Starting...")

    # Start browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("http://localhost:5000")

    time.sleep(1.5)  # allow page to load

    # -----------------------------
    # Read target angle
    # -----------------------------
    target = driver.find_element(By.ID, "target-arrow")
    style = target.get_attribute("style")

    print("Raw target style:", style)

    # Extract rotate(xdeg)
    match = re.search(r'rotate\(([-\d]+)deg\)', style)
    if not match:
        raise ValueError("Could not parse angle from style:", style)

    angle = int(match.group(1))
    print("Parsed target angle:", angle)

    # -----------------------------
    # Rotate user arrow
    # -----------------------------
    user_arrow = driver.find_element(By.ID, "user-arrow")

    # Update browser element
    driver.execute_script(
        f"arguments[0].style.transform='rotate({angle}deg)'",
        user_arrow
    )

    # Update internal JS angle variable so CAPTCHA logic reads correctly
    driver.execute_script(f"currentAngle = {angle}")

    print("Bot rotated user arrow (visual + JS variable updated).")

    # -----------------------------
    # Click submit
    # -----------------------------
    submit_btn = driver.find_element(By.TAG_NAME, "button")
    time.sleep(0.7)  # look human
    submit_btn.click()

    print("Bot submitted form.")

    # -----------------------------
    # Read result
    # -----------------------------
    time.sleep(1)
    status = driver.find_element(By.ID, "status").text
    print("CAPTCHA RESULT →", status)

    # -----------------------------
    # Keep browser open
    # -----------------------------
    print("\nThe browser will remain open. Press ENTER to close it.")
    input()

    driver.quit()


if __name__ == "__main__":
    solve_captcha()
