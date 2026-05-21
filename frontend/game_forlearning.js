// ================================================================
//  CONFIG
// ================================================================
const MODEL_PATH     = "./Models/model_newone_r.json";   // ← path สัมพัทธ์ ใช้ได้ใน browser
const HOLD_FRAMES    = 20;   // frame ที่ต้องถือค้างก่อน confirm
const CONF_THRESHOLD = 0.7;  // confidence ขั้นต่ำ

// A–Y ยกเว้น J, Z (dataset ไม่มี motion)
const LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

// ================================================================
//  STATE
// ================================================================
let model      = null;
let score      = 0;
let roundNum   = 1;
let curWord    = null;
let curSlot    = 0;
let holdCount  = 0;
let lastLetter = null;
let gameActive = false;

// ================================================================
//  DOM refs
// ================================================================
const video   = document.getElementById("video");
const canvas  = document.getElementById("overlay-canvas");
const ctx     = canvas.getContext("2d");
const slotsEl = document.getElementById("word-slots");
const hintEl  = document.getElementById("hint-text");
const msgEl   = document.getElementById("message");
const predEl  = document.getElementById("pred-letter");
const confEl  = document.getElementById("pred-conf");
const holdBar = document.getElementById("hold-bar");
const camDot  = document.getElementById("cam-dot");
const mpDot   = document.getElementById("mp-dot");
const nnDot   = document.getElementById("nn-dot");
const scoreEl = document.getElementById("score");
const roundEl = document.getElementById("round");
const noModel = document.getElementById("no-model-warn");

// ================================================================
//  LOAD MODEL
// ================================================================
async function loadModel() {
  try {
    model = await tf.loadLayersModel(MODEL_PATH);
    nnDot.className = "dot on";
  } catch(e) {
    nnDot.className = "dot err";
    noModel.style.display = "block";
    console.warn("model not found:", e);
  }
}

// ================================================================
//  GAME LOGIC
// ================================================================
function pickWord() {
  const entry = WORDS[Math.floor(Math.random() * WORDS.length)];
  curWord    = entry.word;
  curSlot    = 0;
  holdCount  = 0;
  gameActive = true;
  hintEl.textContent = entry.hint;   // textContent คือเปลี่ยนแค่ข้อความข้างใน โดยมาจาก entry.hint
  msgEl.textContent  = "";
  msgEl.className    = "";
  renderSlots();
  roundEl.textContent = roundNum;
}

function renderSlots() {
  slotsEl.innerHTML = "";
  for (let i = 0; i < curWord.length; i++) {
    const d = document.createElement("div");
    d.className = "slot" + (i === curSlot ? " active" : "");
    d.id = "slot-" + i;
    slotsEl.appendChild(d);
  }
}

function confirmLetter(letter) {
  if (!gameActive || curSlot >= curWord.length) return;

  const expected = curWord[curSlot];
  const slotEl   = document.getElementById("slot-" + curSlot);

  if (letter === expected) {
    slotEl.textContent = letter;
    slotEl.className   = "slot filled";
    curSlot++;

    if (curSlot === curWord.length) {
      score++;
      scoreEl.textContent = score;
      msgEl.textContent   = "🎉 Correct! Well done!";
      msgEl.className     = "win";
      gameActive = false;
    } else {
      document.getElementById("slot-" + curSlot).className = "slot active";
    }
  } else {
    // ผิด → shake แล้วรีเซ็ตทุก slot
    slotEl.className = "slot wrong";
    setTimeout(() => {
      curSlot   = 0;
      holdCount = 0;
      msgEl.textContent = `✗ Wrong! Got "${letter}" . Try again from the start.`;
      msgEl.className   = "lose";
      for (let i = 0; i < curWord.length; i++) {
        const s = document.getElementById("slot-" + i);
        if (s) { s.textContent = ""; s.className = "slot" + (i === 0 ? " active" : ""); }
      }
    }, 600);
  }

  holdCount = 0;
}

// ================================================================
//  INFERENCE
// ================================================================
function predict(landmarks) {
  if (!model) return null;
  const flat = [];
  // โมเดลรับ 63 ค่า = 21 จุด x (x, y, z)
  for (const pt of landmarks) flat.push(pt.x, pt.y, pt.z);
  const t    = tf.tensor2d([flat]);
  const out  = model.predict(t);
  const probs = out.dataSync();
  t.dispose(); out.dispose();

  const idx  = probs.indexOf(Math.max(...probs));
  const conf = probs[idx];
  return { letter: LABELS[idx], conf };
}

// ================================================================
//  DRAW LANDMARKS
// ================================================================
function drawLandmarks(lm) {
  const W = canvas.width, H = canvas.height;
  const CONN = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
  ];

  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(124,106,247,0.7)";
  ctx.lineWidth   = 2;

  for (const [a,b] of CONN) {
    ctx.beginPath();
    ctx.moveTo(lm[a].x * W, lm[a].y * H);
    ctx.lineTo(lm[b].x * W, lm[b].y * H);
    ctx.stroke();
  }
  for (const pt of lm) {
    ctx.beginPath();
    ctx.arc(pt.x * W, pt.y * H, 4, 0, 2*Math.PI);
    ctx.fillStyle = "#4ecdc4";
    ctx.fill();
  }
}

// ================================================================
//  MEDIAPIPE
// ================================================================
const handsModel = new Hands({ locateFile: f =>
  `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });

handsModel.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.5
});

handsModel.onResults(res => {
  canvas.width  = video.videoWidth  || canvas.offsetWidth;
  canvas.height = video.videoHeight || canvas.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  mpDot.className = "dot on";

  if (!res.multiHandLandmarks || res.multiHandLandmarks.length === 0) {
    predEl.textContent = "—";
    confEl.textContent = "";
    holdCount  = 0;
    holdBar.style.width = "0%";
    lastLetter = null;
    return;
  }

  const lm = res.multiHandLandmarks[0];
  drawLandmarks(lm);

  const result = predict(lm);
  if (!result) return;

  const { letter, conf } = result;
  predEl.textContent = conf >= CONF_THRESHOLD ? letter : "—";
  confEl.textContent = `${(conf * 100).toFixed(0)}%`;

  if (conf >= CONF_THRESHOLD && gameActive) {
    if (letter === lastLetter) {
      holdCount++;
    } else {
      holdCount  = 0;
      lastLetter = letter;
    }

    holdBar.style.width = Math.min(holdCount / HOLD_FRAMES * 100, 100) + "%";

    if (holdCount >= HOLD_FRAMES) {
      holdCount = 0;
      holdBar.style.width = "0%";
      confirmLetter(letter);
    }
  } else {
    holdCount = 0;
    holdBar.style.width = "0%";
  }
});

// ================================================================
//  CAMERA
// ================================================================
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
    camDot.className = "dot on";

    const cam = new Camera(video, {
      onFrame: async () => { await handsModel.send({ image: video }); },
      width: 640, height: 480
    });
    cam.start();
  } catch(e) {
    camDot.className = "dot err";
    console.error("camera error:", e);
  }
}

// ================================================================
//  BUTTONS
// ================================================================
document.getElementById("btn-next").onclick = () => {
  roundNum++;
  pickWord();
};
document.getElementById("btn-skip").onclick = () => {
  roundNum++;
  msgEl.textContent = `Skipped — the word was "${curWord}"`;
  msgEl.className   = "lose";
  gameActive = false;
};

// ================================================================
//  INIT
// ================================================================
(async () => {
  await loadModel();
  await startCamera();
  pickWord();
})();
