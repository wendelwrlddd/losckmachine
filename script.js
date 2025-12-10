/**
 * Face Analysis System - Advanced Implementation
 * Uses TensorFlow.js and MediaPipe Face Mesh
 */

// --- Configuration ---
const CONFIG = {
    videoWidth: 640,
    videoHeight: 480,
    analysisInterval: 10, // Run expensive pixel analysis every N frames
    thresholds: {
        oiliness: 200, // Pixel brightness threshold (0-255)
        beardDarkness: 100, // Pixel darkness for beard
    }
};

// --- State ---
let model = null;
let isHeatmapEnabled = false;
let isModelLoaded = false;
let rafId = null;
let frameCount = 0;

// Offscreen canvas for pixel reading
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = CONFIG.videoWidth;
offscreenCanvas.height = CONFIG.videoHeight;
const offCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const finalCtx = ctx; // Alias
const statusText = document.getElementById('status-text');
const statusDot = document.querySelector('.dot');

// Metrics State
let analysisState = {
    symmetry: 0,
    texture: 0, // 0 (Smooth) to 100 (Rough)
    oiliness: 0, // 0 (Matte) to 100 (Oily)
    beardDensity: 0, // 0 (Clean shaven) to 100 (Full beard)
    lastUpdated: 0
};

// --- Landmark Indices (Approximate for MediaPipe 468) ---
const ROI = {
    forehead: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152], // Broad loop, we'll pick center
    foreheadPoly: [10, 297, 332, 284, 251, 21], // Simpler poly
    leftCheek: [123, 50, 205, 117, 118, 119, 120],
    rightCheek: [352, 280, 425, 346, 347, 348, 349],
    chin: [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234], // Lower contour
    jawArea: [0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 185, 61, 291] // Mouth/Chin area
};

// --- Initialization ---
async function init() {
    try {
        await setupCamera();
        await loadModels();
        detectFace();
        setupEvents();
    } catch (error) {
        console.error("Initialization error:", error);
        statusText.innerText = "Erro: " + error.message;
        statusDot.style.backgroundColor = "red";
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: CONFIG.videoWidth, height: CONFIG.videoHeight, facingMode: 'user' },
        audio: false
    });
    video.srcObject = stream;
    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve(video);
        };
    });
}

function setupEvents() {
    document.getElementById('toggle-heatmap').addEventListener('click', () => {
        isHeatmapEnabled = !isHeatmapEnabled;
        const btn = document.getElementById('toggle-heatmap');
        btn.classList.toggle('btn-primary', !isHeatmapEnabled);
        btn.classList.toggle('btn-secondary', isHeatmapEnabled); // Toggle visual style
        btn.innerText = isHeatmapEnabled ? "Ocultar Heatmap" : "Ver Heatmap";
    });
    
    document.getElementById('reset-view').addEventListener('click', () => {
        isHeatmapEnabled = false;
        document.getElementById('toggle-heatmap').innerText = "Ver Heatmap";
    });

    // Mobile: Toggle Sidebar on Click/Tap
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }
}

async function loadModels() {
    statusText.innerText = "Carregando modelos IA...";
    statusDot.classList.add('loading');
    
    // Load detector
    model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
        { maxFaces: 1 }
    );
    
    isModelLoaded = true;
    statusText.innerText = "Monitorando";
    statusDot.classList.remove('loading');
    statusDot.style.backgroundColor = "#10b981";
}

// --- Main Loop ---
async function detectFace() {
    // 1. Draw Video to Offscreen Canvas for Pixel Access (Mirrored for consistency)
    offCtx.save();
    offCtx.scale(-1, 1);
    offCtx.drawImage(video, -CONFIG.videoWidth, 0, CONFIG.videoWidth, CONFIG.videoHeight);
    offCtx.restore();

    // 2. Clear Main Canvas
    finalCtx.clearRect(0, 0, canvas.width, canvas.height);

    if (isModelLoaded) {
        const predictions = await model.estimateFaces({
            input: video,
            flipHorizontal: true // Flip predictions to match mirrored video
        });

        if (predictions.length > 0) {
            const face = predictions[0];
            
            // 3. Run Analysis (throttled)
            frameCount++;
            if (frameCount % CONFIG.analysisInterval === 0) {
                analyzeFaceFeatures(face, offCtx);
                updateUI();
            }

            // 4. Draw Visuals
            if (isHeatmapEnabled) {
                drawHeatmap(face, finalCtx);
            } else {
                drawLandmarks(face, finalCtx);
            }
        }
    }
    
    rafId = requestAnimationFrame(detectFace);
}

// --- Analysis Algorithms ---
function analyzeFaceFeatures(face, ctxSource) {
    const landmarks = face.scaledMesh;

    // 1. Symmetry
    // Nose Top: 6, Nose Tip: 1
    // Left Cheek: 234, Right Cheek: 454
    // Left Eye Outer: 33, Right Eye Outer: 263
    const nose = landmarks[1];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];

    const dCheekL = euclideanDist(nose, leftCheek);
    const dCheekR = euclideanDist(nose, rightCheek);
    const dEyeL = euclideanDist(nose, leftEye);
    const dEyeR = euclideanDist(nose, rightEye);

    const symCheek = Math.min(dCheekL, dCheekR) / Math.max(dCheekL, dCheekR);
    const symEye = Math.min(dEyeL, dEyeR) / Math.max(dEyeL, dEyeR);
    
    analysisState.symmetry = Math.round(((symCheek + symEye) / 2) * 100);

    // 2. Skin Texture & Oiliness (Sampling Regions)
    // Extract ROI Pixels
    const cheekData = getRegionPixels(ctxSource, landmarks, ROI.leftCheek);
    const foreheadData = getRegionPixels(ctxSource, landmarks, ROI.foreheadPoly);
    
    // Texture: StdDev of Grayscale (rough approximation)
    const textureScore = calculateTextureMetric(cheekData);
    analysisState.texture = Math.min(100, Math.round(textureScore * 2)); // Scale factor

    // Oiliness: % of bright pixels in forehead
    const oilScore = calculateOilinessMetric(foreheadData);
    analysisState.oiliness = Math.min(100, Math.round(oilScore));

    // 3. Beard Analysis
    // Compare lower face density/color to upper face
    // If lower face is significantly darker or has high high-freq noise compared to cheek
    const chinData = getRegionPixels(ctxSource, landmarks, [2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18]); // Jawline strip
    const beardScore = calculateBeardMetric(chinData, cheekData);
    analysisState.beardDensity = Math.min(100, Math.round(beardScore));
}

// --- Helpers for Image Processing ---

function getRegionPixels(ctx, landmarks, indices) {
    // 1. Find Bounding Box for the ROI
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    indices.forEach(idx => {
        const pt = landmarks[idx];
        if (pt[0] < minX) minX = pt[0];
        if (pt[1] < minY) minY = pt[1];
        if (pt[0] > maxX) maxX = pt[0];
        if (pt[1] > maxY) maxY = pt[1];
    });

    // Clamp
    minX = Math.max(0, minX); minY = Math.max(0, minY);
    maxX = Math.min(CONFIG.videoWidth, maxX); maxY = Math.min(CONFIG.videoHeight, maxY);
    const w = maxX - minX;
    const h = maxY - minY;

    if (w <= 0 || h <= 0) return null;

    return ctx.getImageData(minX, minY, w, h);
}

function calculateTextureMetric(imageData) {
    if (!imageData) return 0;
    const data = imageData.data;
    let sum = 0;
    let sqSum = 0;
    let count = 0;

    // Convert to grayscale and calc variance
    for (let i = 0; i < data.length; i += 4) {
        const brightness = (data[i] + data[i+1] + data[i+2]) / 3;
        sum += brightness;
        sqSum += brightness * brightness;
        count++;
    }

    const mean = sum / count;
    const variance = (sqSum / count) - (mean * mean);
    const stdDev = Math.sqrt(variance);
    
    return stdDev; // Higher stdDev = more contrasty details (roughness/hair/pores)
}

function calculateOilinessMetric(imageData) {
    if (!imageData) return 0;
    const data = imageData.data;
    let shinyPixels = 0;
    let totalPixels = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
        // Simple heuristic: High brightness
        const brightness = (data[i] + data[i+1] + data[i+2]) / 3;
        if (brightness > 190) { // Threshold for "shine"
            shinyPixels++;
        }
    }
    
    // Percentage
    return (shinyPixels / totalPixels) * 100 * 3; // Boosted factor
}

function calculateBeardMetric(chinData, cheekData) {
    if (!chinData || !cheekData) return 0;
    
    // Compare mean brightness
    const getMean = (img) => {
        let s = 0; 
        for(let i=0; i<img.data.length; i+=4) s += (img.data[i]+img.data[i+1]+img.data[i+2])/3;
        return s / (img.data.length/4);
    };

    const chinMean = getMean(chinData);
    const cheekMean = getMean(cheekData);

    // If chin is darker than cheeks, likely beard
    const diff = cheekMean - chinMean;
    if (diff > 0) {
        return Math.min(100, diff * 2); 
    }
    return 0; // Chin is brighter or same? No beard
}

function euclideanDist(p1, p2) {
    return Math.sqrt(Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2));
}

// --- Drawing ---

function drawLandmarks(face, ctx) {
    ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
    face.scaledMesh.forEach((pt, i) => {
        // Draw T-Zone and Cheeks more prominently
        if (ROI.foreheadPoly.includes(i) || ROI.leftCheek.includes(i) || ROI.rightCheek.includes(i)) {
             ctx.fillStyle = 'rgba(79, 70, 229, 0.8)';
             ctx.fillRect(pt[0], pt[1], 2, 2);
        } else if (i % 6 === 0) {
             ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
             ctx.fillRect(pt[0], pt[1], 1, 1);
        }
    });
}

function drawHeatmap(face, ctx) {
    const landmarks = face.scaledMesh;
    
    // Function to draw simple gradient blob
    const drawBlob = (indices, color) => {
        // Find center of indices
        let cx=0, cy=0;
        indices.forEach(i => { cx += landmarks[i][0]; cy += landmarks[i][1]; });
        cx /= indices.length;
        cy /= indices.length;

        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
        grad.addColorStop(0, color);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 60, 0, 2*Math.PI);
        ctx.fill();
    };

    // Draw Heatmap layers
    // 1. Oiliness (Forehead) -> Yellow/White
    if (analysisState.oiliness > 30) {
        drawBlob(ROI.foreheadPoly, `rgba(255, 255, 0, ${analysisState.oiliness / 150})`);
    }

    // 2. Texture (Cheeks) -> Red
    if (analysisState.texture > 15) {
        drawBlob(ROI.leftCheek, `rgba(255, 50, 50, ${analysisState.texture / 150})`);
        drawBlob(ROI.rightCheek, `rgba(255, 50, 50, ${analysisState.texture / 150})`);
    }

    // 3. Beard -> Blue
    if (analysisState.beardDensity > 20) {
        drawBlob(ROI.chin, `rgba(50, 50, 255, ${analysisState.beardDensity / 150})`);
    }
}

// --- UI Updates ---
function updateUI() {
    updateBar('symmetry', analysisState.symmetry);
    updateBar('texture', analysisState.texture);
    updateBar('oiliness', analysisState.oiliness);
    updateBar('beard', analysisState.beardDensity);

    updateInsights();
}

function updateBar(id, val) {
    const el = document.getElementById(`score-${id}`);
    const bar = document.getElementById(`bar-${id}`);
    if(el) el.innerText = `${val}%`;
    if(bar) bar.style.width = `${val}%`;
}

function updateInsights() {
    const list = document.getElementById('insights-list');
    const box = document.getElementById('suggestion-box');
    let items = "";
    let suggestion = "";

    // Symmetria
    if (analysisState.symmetry > 90) items += "<li>✅ Rosto altamente simétrico.</li>";
    else if (analysisState.symmetry < 80) items += "<li>⚠️ Leve assimetria detectada.</li>";

    // Pele
    if (analysisState.texture > 30) {
        items += "<li>🔍 Textura irregular detectada.</li>";
        suggestion += "Considere esfoliação suave e hidratação. ";
    } else {
        items += "<li>✨ Pele com aparência uniforme.</li>";
    }

    // Oleosidade
    if (analysisState.oiliness > 50) {
        items += "<li>💧 Alta refletividade na zona T.</li>";
        suggestion += "Uso de sabonete oil-control recomendado. ";
    }

    // Barba
    if (analysisState.beardDensity > 40) {
        items += "<li>🧔 Barba densa identificada.</li>";
        
        // Formato rosto suggestion based on symmetry/jaw
        if (analysisState.symmetry > 85) suggestion += "Estilo 'Boxed Beard' valorizaria sua simetria. ";
        else suggestion += "Deixe a barba crescer nas laterais para equilibrar. ";
    } else {
        items += "<li>Mantenha a pele hidratada pós-barbear.</li>";
    }

    list.innerHTML = items;
    if (suggestion) box.innerHTML = `<p>${suggestion}</p>`;
    else box.innerHTML = "<p>Você está com ótima aparência! Mantenha sua rotina.</p>";
}

// Boot
window.onload = init;
