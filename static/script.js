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

// ----- AI Detection Helpers -----

// Path complexity
function pathComplexity(path) {
    if(path.length < 2) return 0;
    let changes = 0;
    for(let i = 1; i < path.length; i++) {
        if(Math.abs(path[i].angle - path[i-1].angle) > 0.5) changes++;
    }
    return changes;
}

// Overshoot count near target
function overshootCount(path, target) {
    let nearTarget = path.filter(p => Math.abs(p.angle - target) <= tolerance);
    let overshoot = 0;
    for(let i = 1; i < nearTarget.length; i++) {
        if(Math.sign(nearTarget[i].angle - target) !== Math.sign(nearTarget[i-1].angle - target))
            overshoot++;
    }
    return overshoot;
}

// Speed variance
function speedVariance(path) {
    if(path.length < 2) return 0;
    let speeds = [];
    for(let i = 1; i < path.length; i++){
        let dt = path[i].t - path[i-1].t;
        let da = path[i].angle - path[i-1].angle;
        speeds.push(Math.abs(da/dt));
    }
    let mean = speeds.reduce((a,b)=>a+b,0)/speeds.length;
    let variance = speeds.reduce((a,b)=>a + Math.pow(b-mean,2),0)/speeds.length;
    return variance;
}

// ----- Submit -----
function sendToServer() {
    const reactionTime = Date.now() - startTime;

    // Check alignment success
    const diff = Math.abs(currentAngle - targetAngle) % 360;
    const distance = diff > 180 ? 360 - diff : diff;
    const aligned = distance <= tolerance;

    // AI detection scoring
    let score = 0;
    if(reactionTime < 200) score += 1;
    if(pathComplexity(movementHistory) < 5) score += 1;
    if(overshootCount(movementHistory, targetAngle) === 0) score += 1;
    if(speedVariance(movementHistory) < 0.01) score += 1;

    const humanThreshold = 2;
    const isHuman = score <= humanThreshold;

    // Show result to user
    if(!aligned) {
        status.innerText = "❌ Failed alignment!";
        status.style.color = "#ff5555";
    } else if(aligned && !isHuman) {
        status.innerText = "⚠️ Likely AI detected!";
        status.style.color = "#ffb74d";
    } else {
        status.innerText = "✅ Success!";
        status.style.color = "#00ff99";

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
        ai_score: score,
        isHuman: isHuman,
        aligned: aligned
    };

    fetch("/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => console.log("Logged:", data));

    // Reset user arrow
    currentAngle = 0;
    userArrow.style.transform = `rotate(0deg)`;
    movementHistory = [];
    startTime = null;
}
