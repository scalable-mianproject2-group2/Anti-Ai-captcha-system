import pyautogui
import time
import random

print("=== PyAutoGUI Slider Automation ===")

# -------------------------------
# Part 1: Coordinate Capture (originally Script 1)
# -------------------------------
print("Step 1: Hover your mouse over the SLIDER HANDLE (just below the image)")
time.sleep(5)  # gives you 5 seconds to move the mouse
start = pyautogui.position()
print("Slider handle START captured at:", start)

print("\nStep 2: Hover your mouse over the TARGET END of the puzzle track")
time.sleep(5)
end = pyautogui.position()
print("Slider TARGET captured at:", end)

# Optional tiny adjustment for sliders just below image
start_y = start.y + 1
end_y = end.y + 1
start_x = start.x
end_x = end.x

# -------------------------------
# Part 2: Drag Slider (originally Script 2)
# -------------------------------
print("\nDragging slider in 3 seconds... switch to browser now!")
time.sleep(3)

# Move to slider handle
pyautogui.moveTo(start_x, start_y, duration=0.5)
pyautogui.mouseDown()

# Smooth human-like drag in small steps
steps = 30
for i in range(steps):
    x = start_x + (end_x - start_x) * (i+1) / steps + random.randint(-1,1)
    y = start_y + random.randint(-1,1)
    pyautogui.moveTo(x, y, duration=0.02)

pyautogui.mouseUp()
print("\nSlider dragged successfully!")
 