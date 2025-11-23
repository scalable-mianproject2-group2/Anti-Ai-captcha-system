// 与 Flutter 版本保持一致的参数
const BOX_W = 300;
const BOX_H = 200;
const PIECE_SIZE = 60;
const MARGIN = 24;
const TOLERANCE = 8; // 容差

let correctX = 0;
let correctY = 0;
let sliderX = 0;
let imgUrl = "";

const bgImage = document.getElementById("bgImage");
const hole = document.getElementById("hole");
const pieceWrapper = document.getElementById("pieceWrapper");
const piece = document.getElementById("piece");
const slider = document.getElementById("slider");
const verifyBtn = document.getElementById("verifyBtn");
const refreshBtn = document.getElementById("refreshBtn");

// 初始化 / 刷新验证码
function initCaptcha() {
  slider.value = 0;
  sliderX = 0;

  // 随机图片，模拟 Flutter 的 NetworkImage(picsum)
  imgUrl =
    "https://picsum.photos/" +
    BOX_W +
    "/" +
    BOX_H +
    "?random=" +
    Math.floor(Math.random() * 999999);

  bgImage.onload = function () {
    // 随机洞口位置，留出边距
    correctX =
      MARGIN + Math.random() * (BOX_W - PIECE_SIZE - 2 * MARGIN);
    correctY =
      MARGIN + Math.random() * (BOX_H - PIECE_SIZE - 2 * MARGIN);

    // 设置洞口位置
    hole.style.left = correctX + "px";
    hole.style.top = correctY + "px";

    // 拼图块初始在最左侧，同一条水平线
    pieceWrapper.style.left = "0px";
    pieceWrapper.style.top = correctY + "px";

    // 拼图块纹理来自同一张图，偏移固定为 -correctX / -correctY
    piece.style.backgroundImage = "url(" + imgUrl + ")";
    piece.style.backgroundSize = BOX_W + "px " + BOX_H + "px";
    piece.style.backgroundPosition =
      -correctX + "px " + -correctY + "px";
  };

  bgImage.src = imgUrl;
}

// slider 拖动事件
slider.addEventListener("input", function (e) {
  sliderX = parseFloat(e.target.value);
  pieceWrapper.style.left = sliderX + "px";
});

// 点击“Verify”
verifyBtn.addEventListener("click", function () {
  const distance = Math.abs(sliderX - correctX);
  const ok = distance <= TOLERANCE;

  if (ok) {
    // 吸附效果
    pieceWrapper.style.left = correctX + "px";
    alert("Verification Successful");
  } else {
    alert("Verification Failed");
  }

  // 把结果发给后端做日志
  fetch("/slider/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      success: ok,
      distance: distance,
      correctX: correctX,
      sliderX: sliderX,
      imgUrl: imgUrl,
    }),
  }).catch((err) => console.error("slider log error:", err));

  // 如果成功，这里跳到音频验证码
  if (ok) {
    window.location.href = "/audio";
  } else {
    // 失败直接刷新当前验证码
    initCaptcha();
  }
});

// 点击 Refresh
refreshBtn.addEventListener("click", function () {
  initCaptcha();
});

// 页面加载完初始化
window.addEventListener("load", function () {
  // slider 最大位置 = BOX_W - PIECE_SIZE
  slider.max = BOX_W - PIECE_SIZE;
  initCaptcha();
});
