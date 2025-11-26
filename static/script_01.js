// Random target angle
const targetAngle = Math.floor(Math.random() * 360); 
let currentAngle = 0;

// Human behavior logs
let startTime = null;
let movementHistory = [];
const tolerance = 10; // degrees for success

// Apply target rotation to target arrow
const targetArrow = document.getElementById("target-arrow");
targetArrow.style.transform = `rotate(${targetAngle}deg)`;
targetArrow.style.setProperty('--angle', `${targetAngle}deg`);

const userArrow = document.getElementById("user-arrow");
const status = document.getElementById("status");

userArrow.addEventListener("mousedown", startDrag);
userArrow.addEventListener("touchstart", startDrag);

function startDrag(e) {
    startTime = Date.now();
    movementHistory = [];

    document.addEventListener("mousemove", rotate);
    document.addEventListener("touchmove", rotate);
    document.addEventListener("mouseup", stopDrag);
    document.addEventListener("touchend", stopDrag);
}

function stopDrag() {
    document.removeEventListener("mousemove", rotate);
    document.removeEventListener("touchmove", rotate);
}

function rotate(e) {
    const rect = userArrow.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - (rect.left + rect.width / 2);
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - (rect.top + rect.height / 2);

    currentAngle = Math.atan2(y, x) * 180 / Math.PI + 90;
    userArrow.style.transform = `rotate(${currentAngle}deg)`;

    movementHistory.push({
        t: Date.now(),
        angle: currentAngle
    });
}

function sendToServer() {
    const reactionTime = Date.now() - startTime;

    // Check success
    const diff = Math.abs(currentAngle - targetAngle) % 360;
    const distance = diff > 180 ? 360 - diff : diff;
    const success = distance <= tolerance;

    // Show result to user
    status.innerText = success ? "✅ Success!" : "❌ Failure!";
    status.style.color = success ? "#00ff99" : "#ff5555";

    // Success glow animation
    if(success) {
        document.querySelector('.captcha-box').classList.add('success-glow');
        setTimeout(() => {
            document.querySelector('.captcha-box').classList.remove('success-glow');
        }, 1000);
    }

    // Send log to backend
    const payload = {
        target_angle: targetAngle,
        final_angle: currentAngle,
        movement_trace: movementHistory,
        reaction_time_ms: reactionTime,
        success: success
    };

    fetch("/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => console.log("Logged:", data));

    // Reset user arrow for next try
    currentAngle = 0;
    userArrow.style.transform = `rotate(0deg)`;
    movementHistory = [];
    startTime = null;
}
