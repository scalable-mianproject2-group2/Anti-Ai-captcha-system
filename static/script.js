// Random target angle
const targetAngle = Math.random() * 360; 
let currentAngle = 0;
let attempts;

// Human behavior logs
let startTime = null;
let movementHistory = [];
const tolerance = 10; // degrees for success

// Apply rotation to target arrow as goal to user to match
const targetArrow = document.getElementById("target-arrow");
targetArrow.style.transform = `rotate(${targetAngle}deg)`;
targetArrow.style.setProperty('--angle', `${targetAngle}deg`);

const userArrow = document.getElementById("user-arrow");
const status = document.getElementById("status");

userArrow.addEventListener("mousedown", startDrag);
userArrow.addEventListener("touchstart", startDrag);
document.addEventListener("mouseup", stopDrag);
document.addEventListener("touchend", stopDrag);

function startDrag(e) {
    startTime = Date.now();
    movementHistory = [];

    document.addEventListener("mousemove", rotate);
    document.addEventListener("touchmove", rotate);
    
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

currentAngle = (currentAngle + 360) % 360;
trunc = Math.trunc(currentAngle);
    console.log(trunc);
    if (trunc >= 600 ){
       userArrow.style.transform = `rotate(${631}deg)`; 
    }
    else{
    userArrow.style.transform = `rotate(${trunc}deg)`;
    }

    movementHistory.push({
        time: Date.now(),
        angle: currentAngle
    });
}

// ----- AI Detection Helpers -----

// Path complexity
function pathComplexity(path) {
    if(path.length < 2) {
        return 0;
    }
    let changes = 0;
    for(let i = 1; i < path.length; i++) {
        if(Math.abs(path[i].angle - path[i-1].angle) > 0.5) {
            changes++;
        }
    }

    return changes;
}

// Overshoot count near target
function overshootCount(path, target) {
    //filters out the path points that are within an angle of the target within the tolerance angle
    let nearTarget = path.filter(p => Math.abs(p.angle - target) <= tolerance);
    let overshoot = 0;
    for (let i = 1; i < nearTarget.length; i++) {
        
        const prevDirection = Math.sign(nearTarget[i-1].angle - target);
        const currDirection = Math.sign(nearTarget[i].angle - target);
        
        // If direction changed (overshot), count it
        if (prevDirection !== currDirection) {
            // the only timg the signs will be different is when  the angle goes past the target eg if the target is 90, 90- 89 is 1 but 90 - 91 is -1
            overshoot++;
        }
    }
    return overshoot;
}

// Speed variance
function speedVariance(path) {
    if (path.length < 2) return 0;
    
    // Calculate speeds between points
    const speeds = [];
    for (let i = 1; i < path.length; i++) {
        const timeDiff = path[i].time - path[i-1].time;
        const angleDiff = Math.abs(path[i].angle - path[i-1].angle);
        speeds.push(angleDiff / timeDiff);
    }
    
    // Calculate average speed
    const avgSpeed = speeds.reduce((sum, speed) => sum + speed, 0) / speeds.length;
    
    // Calculate how much each speed differs from average
    const variance = speeds.reduce((sum, speed) => 
        sum + Math.pow(speed - avgSpeed, 2), 0) / speeds.length;
    
    return variance;
}

// ----- Submit -----
function sendToServer() {
    const reactionTime = Date.now() - startTime;

    // Check alignment success
    const diff = Math.abs(currentAngle - targetAngle) % 360;
    let distance;
        if (diff > 180) {
             distance = 360 - diff;  // Take the shorter path around circle
                }
       else {
            distance = diff;        // Use direct difference
            }
     let aligned;
    if (distance <= tolerance){
         aligned = true;
    }
    else {
        aligned = false;
    }

    // AI detection scoring
    let score = 0;
    if(reactionTime < 200) score += 1;
    if(pathComplexity(movementHistory) < 5) score += 1;
    if(overshootCount(movementHistory, targetAngle) === 0) score += 1;
    if(speedVariance(movementHistory) < 0.01) score += 1;

    const humanThreshold = 2;
    
    if (score <= humanThreshold){
        var isHuman = true;
    }
    else {
        var isHuman = false;
    }

    // Show result to user
    if(!aligned) {
        status.innerText = "Failed alignment!";
        status.style.color = "#ff5555";
       attempts = parseInt(localStorage.getItem('failedAttempts') || '0');
        localStorage.setItem('failedAttempts', (attempts + 1).toString());

    } else if(aligned && !isHuman) {
        status.innerText = "Likely AI detected!";
        status.style.color = "#ffb74d";
        
    } else {
        attempts = parseInt(localStorage.getItem('failedAttempts') || '0');
        if (attempts >= 5) {
            status.innerText = "Too many failed attempts previously!";
            status.style.color = "#ff5555";
            
        }
        else {
        status.innerText = "Success!";
        status.style.color = "#00ff99";
        localStorage.setItem('failedAttempts', (0).toString());
        

        document.querySelector('.captcha-box').classList.add('success-glow');
        setTimeout(() => {
            document.querySelector('.captcha-box').classList.remove('success-glow');
        }, 1000);
        
    }
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
