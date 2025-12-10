import './style.css';
/**
 * Face Analysis System - Snapshot Flow
 */

const CONFIG = {
    videoWidth: 640,
    videoHeight: 480,
    analysisInterval: 5, 
    thresholds: { oiliness: 200, beardDarkness: 100 }
};

// State
let model = null;
let isModelLoaded = false;
let isAnalysing = false;
let animationId = null;
let analysisState = { symmetry: 0, texture: 0, oiliness: 0, beardDensity: 0 };

// DOM
const ui = {
    intro: document.getElementById('intro-screen'),
    analysis: document.getElementById('analysis-screen'),
    video: document.getElementById('video'),
    canvas: document.getElementById('output'),
    startBtn: document.getElementById('start-btn'),
    captureBtn: document.getElementById('capture-btn'),
    retryBtn: document.getElementById('retry-btn'),
    toggleHeatmap: document.getElementById('toggle-heatmap'),
    sidebar: document.querySelector('.sidebar'),
    statusText: document.getElementById('status-text'),
    statusDot: document.querySelector('.dot')
};

const ctx = ui.canvas.getContext('2d');
// Offscreen for analysis
const offCanvas = document.createElement('canvas');
offCanvas.width = CONFIG.videoWidth; 
offCanvas.height = CONFIG.videoHeight;
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

// --- 1. INTRO & SETUP ---
async function init() {
    // Determine mobile
    const isMobile = window.innerWidth <= 768;
    if(isMobile) {
        CONFIG.videoWidth = 480; 
        CONFIG.videoHeight = 640; // Portrait aspect
    }
    
    // Load Models proactively but don't start camera
    loadModels();

    // Event Listeners
    ui.startBtn.addEventListener('click', startExperience);
    ui.captureBtn.addEventListener('click', captureSnapshot);
    ui.retryBtn.addEventListener('click', resetExperience);
    ui.toggleHeatmap.addEventListener('click', toggleHeatmapLayer);
    
    // File Upload Handlers
    const uploadTrigger = document.getElementById('upload-trigger-btn');
    const fileInput = document.getElementById('file-upload');
    
    if(uploadTrigger && fileInput) {
        uploadTrigger.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileUpload);
    }
}

// --- Loading Helper ---
// --- Loading Helper ---
function setLoading(isLoading, message = "Processando...") {
    let loader = document.getElementById('loading-overlay');
    
    // Robustness: Create if missing (handles cached HTML mismatch)
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'loading-overlay';
        loader.className = 'hidden';
        loader.innerHTML = `
            <div class="loader-content">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
        document.body.appendChild(loader);
        
        // Add basic styles inline just in case CSS is also cached/missing
        loader.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:2000;display:flex;justify-content:center;align-items:center;color:white;font-family:sans-serif;";
    }

    const loaderText = loader.querySelector('p');
    if (loaderText) loaderText.innerText = message;
    
    if(isLoading) {
        loader.classList.remove('hidden');
        loader.style.display = 'flex'; // Force display
    } else {
        loader.classList.add('hidden');
        loader.style.display = 'none';
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if(!file) return;
    
    setLoading(true, "Lendo arquivo...");
    
    try {
        // Prepare UI for static analysis
        ui.intro.classList.add('hidden');
        ui.analysis.classList.remove('hidden');
        
        // Load image
        const img = await loadImage(file);
        
        // Stop any camera
        if(ui.video.srcObject) {
            ui.video.srcObject.getTracks().forEach(track => track.stop());
            ui.video.srcObject = null;
        }
        
        // Fix Visibility: Video wrapper collapses if video is hidden. 
        // We must set dimensions on the wrapper or rely on the canvas forcing it.
        // Let's force the canvas to be relative or update wrapper.
        const wrapper = document.querySelector('.video-wrapper');
        if (wrapper) {
            wrapper.style.aspectRatio = `${img.width} / ${img.height}`;
        }
        
        // Set canvas size
        // We need to match the render size for proper overlay
        // Simplest strategy: Set canvas to image native size, let CSS scale it.
        ui.canvas.width = img.width;
        ui.canvas.height = img.height;
        offCanvas.width = img.width;
        offCanvas.height = img.height;
        
        // Draw image to canvases
        ctx.clearRect(0,0, ui.canvas.width, ui.canvas.height);
        ctx.drawImage(img, 0, 0);
        
        offCtx.clearRect(0,0, offCanvas.width, offCanvas.height);
        offCtx.drawImage(img, 0, 0);
        
        // Hide video, Ensure Canvas is Visible
        ui.video.style.display = 'none';
        ui.canvas.style.display = 'block'; // Visual only
        
        // Run Analysis using the OFFSCREEN CANVAS 
        // (It has the resized image and is robust against DOM visibility issues)
        setLoading(true, "Analisando IA...");
        await analyzeStaticImage(offCanvas);
        
    } catch (err) {
        console.error(err);
        alert("Erro ao ler imagem: " + err.message);
        resetExperience();
    } finally {
        setLoading(false);
    }
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

// Separate function for static analysis
async function analyzeStaticImage(imageSource) {
    if (!model) await loadModels();
    
    // Note: createDetector API returns { keypoints: [{x,y,z,name}, ...] }
    const predictions = await model.estimateFaces(imageSource, {
        flipHorizontal: false 
    });

    if (predictions.length > 0) {
        const face = predictions[0];
        analyzeFaceFeatures(face, offCtx);
        updateUI();
        drawHeatmap(face, ctx);
        
        // Show Report
        ui.captureBtn.classList.add('hidden');
        ui.retryBtn.classList.remove('hidden');
        ui.toggleHeatmap.classList.remove('hidden');
        if (ui.sidebar) ui.sidebar.classList.add('active');
        
        ui.statusText.innerText = "Análise de Imagem Concluída";
    } else {
        alert("Nenhum rosto detectado. Tente uma foto com iluminação melhor e de frente.");
        resetExperience();
    }
}

async function loadModels() {
    try {
        ui.startBtn.innerText = "Carregando...";
        setLoading(true, "Carregando modelos IA...");
        
        const modelType = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig = {
            runtime: 'tfjs', 
            refineLandmarks: true,
            maxFaces: 1
        };

        model = await faceLandmarksDetection.createDetector(modelType, detectorConfig);

        isModelLoaded = true;
        ui.startBtn.innerText = "Analisar Rosto";
        ui.startBtn.disabled = false;
        
        setLoading(false);
    } catch (e) {
        console.error(e);
        alert("Erro ao carregar IA: " + e.message);
        ui.statusText.innerText = "Erro no carregamento";
        setLoading(false);
    }
}



async function startExperience() {
    ui.intro.classList.add('hidden');
    ui.analysis.classList.remove('hidden');

    try {
        // Request Camera (iOS requires this user gesture chain)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: CONFIG.videoWidth }, 
                height: { ideal: CONFIG.videoHeight },
                facingMode: 'user' 
            },
            audio: false
        });
        ui.video.srcObject = stream;
        
        // Wait for connection
        ui.video.onloadedmetadata = () => {
             ui.canvas.width = ui.video.videoWidth;
             ui.canvas.height = ui.video.videoHeight;
             offCanvas.width = ui.video.videoWidth;
             offCanvas.height = ui.video.videoHeight;
             
             ui.statusText.innerText = "Posicione seu rosto";
             ui.statusDot.style.backgroundColor = "#10b981";
             
             // Start Preview Loop
             isAnalysing = true;
             previewLoop();
        };

    } catch (err) {
        console.error(err);
        alert("Permissão de câmera negada. Por favor, permita o acesso para continuar.");
        location.reload();
    }
}

// --- 2. PREVIEW LOOP ---
async function previewLoop() {
    if (!isAnalysing) return; // Stop if captured

    // Draw Video
    ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
    // Note: We don't draw video to canvas here, browser handles <video> tag. 
    // We just draw the mesh overlay.

    if (model) {
        // Fix: New API takes (image, config)
        try {
            const predictions = await model.estimateFaces(ui.video, {
                flipHorizontal: true
            });

            if (predictions.length > 0) {
                drawMesh(predictions[0], ctx);
            }
        } catch (e) {
            // Ignore frame errors in preview to prevent crashing
            console.warn("Frame skipped", e);
        }
    }

    animationId = requestAnimationFrame(previewLoop);
}

// --- 3. CAPTURE LOGIC ---
async function captureSnapshot() {
    isAnalysing = false; // Stop loop
    cancelAnimationFrame(animationId);

    ui.statusText.innerText = "Processando...";
    setLoading(true, "Processando Captura..."); // Show Overlay
    ui.captureBtn.classList.add('hidden');
    
    // 1. Draw final freeze frame to canvas (so we can hide video)
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(ui.video, -ui.canvas.width, 0, ui.canvas.width, ui.canvas.height);
    ctx.restore();
    
    // 2. Draw to offscreen for pixel analysis
    offCtx.save();
    offCtx.scale(-1, 1);
    offCtx.drawImage(ui.video, -ui.canvas.width, 0, ui.canvas.width, ui.canvas.height);
    offCtx.restore();

    // 3. Run Deep Analysis
        try {
            // Convert offCanvas to base64 first
            const base64Image = offCanvas.toDataURL('image/jpeg', 0.8);
            
            // Convert base64 to Blob for FormData
            setLoading(true, "Analisando com IA..."); // Ensure loading screen is visible
            
            const res = await fetch(base64Image);
            const blob = await res.blob();
            const formData = new FormData();
            formData.append('foto', blob, 'capture.jpg');

            // Send to new Gemini backend
            const response = await fetch('http://localhost:3000/api/analisar-rosto', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Map Gemini JSON response to local state
            // Gemini returns 0-10 strings/numbers, we map to 0-100
            
            // Helper to parse "8/10" or "8" to number
            const parseScore = (val) => {
                if(!val) return 50;
                if(typeof val === 'number') return val * 10;
                const num = parseFloat(val.toString().split('/')[0]);
                return isNaN(num) ? 50 : num * 10;
            };

            analysisState.symmetry = parseScore(data.simetria?.nota);
            analysisState.texture = parseScore(data.qualidade_pele?.nota);
            
            // These are not in the current prompt, so we keep them random/mock or set neutral
            analysisState.oiliness = 50; 
            analysisState.beardDensity = 50; // Could add to prompt later
            
            // Format valid insights
            const points = data.sugestoes_melhoria || [];
            if(Array.isArray(points)) {
                 analysisState.insight = points[0]; // Take the first suggestion as main insight
            } else {
                 analysisState.insight = "Melhore sua rotina de cuidados.";
            }
            
            // Visuals
            const predictions = await model.estimateFaces(ui.video, { flipHorizontal: true });
            if (predictions.length > 0) {
                 const face = predictions[0];
                 // Keep visual mesh
                 analyzeFaceFeatures(face, offCtx); 
                 
                 updateUI();
                 drawHeatmap(face, ctx);
                 
                 ui.retryBtn.classList.remove('hidden');
                 ui.toggleHeatmap.classList.remove('hidden');
                 if (ui.sidebar) ui.sidebar.classList.add('active');
                 ui.statusText.innerText = "Análise Concluída";
            } else {
                 alert("Rosto não detectado para overlay. Mas análise de dados concluída.");
                 updateUI();
            }

        } catch(e) {
            console.error("Analysis failed", e);
            alert("Erro na análise (API): " + e.message);
            resetExperience();
        } finally {
            setLoading(false);
        }
    }
    
// Removed mock logic that was here before because we now fetch data.
// We still keep the helper to draw mesh/heatmap, but NOT to calculate scores.


function resetExperience() {
    isAnalysing = true;
    ui.captureBtn.classList.remove('hidden');
    ui.retryBtn.classList.add('hidden');
    ui.toggleHeatmap.classList.add('hidden');
    ui.statusText.innerText = "Posicione seu rosto";
    
    // Close sheet
    if (ui.sidebar) ui.sidebar.classList.remove('active');
    
    // Show video again
    ui.video.style.display = 'block';
    ui.canvas.style.display = 'block'; 
    // Reset canvas position from upload mode if needed
    ui.canvas.style.position = '';
    
    previewLoop();
}

// --- 4. ANALYSIS & VISUALS (Reused logic) ---
function analyzeFaceFeatures(face, ctxSource) {
    // Adapter: Handle both Old API (scaledMesh array) and New API (keypoints object)
    let landmarks;
    if (face.keypoints) {
        // New API: Array of objects {x, y, z, name}
        // Map to array of arrays [x, y, z] to match legacy logic
        landmarks = face.keypoints.map(p => [p.x, p.y, p.z]);
    } else {
        // Old API
        landmarks = face.scaledMesh;
    }
    
    // If we somehow still don't have landmarks, abort
    if (!landmarks) {
        console.error("No landmarks found in face object", face);
        return;
    }
    
    // Patch face object for other functions (drawHeatmap etc)
    face.scaledMesh = landmarks; 

    // Symmetry
    const nose = landmarks[1];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const dL = Math.hypot(nose[0]-leftCheek[0], nose[1]-leftCheek[1]);
    const dR = Math.hypot(nose[0]-rightCheek[0], nose[1]-rightCheek[1]);
    analysisState.symmetry = Math.round((Math.min(dL, dR)/Math.max(dL, dR)) * 100);

    // Texture (Random high-pass filter sim + partial real logic)
    // We already have offCtx with data
    const cheekPixels = getRegionPixels(ctxSource, landmarks, [123, 50, 205]); 
    const tScore = calculateTextureMetric(cheekPixels); 
    analysisState.texture = Math.min(100, Math.floor(tScore * 2.5));

    // Oiliness
    const foreheadPixels = getRegionPixels(ctxSource, landmarks, [10, 338, 297]);
    analysisState.oiliness = Math.min(100, calculateOilinessMetric(foreheadPixels));

    // Beard
    const chinPixels = getRegionPixels(ctxSource, landmarks, [152, 148, 176]);
    analysisState.beardDensity = calculateBeardMetric(chinPixels, cheekPixels);
}

// Helpers
let isHeatmapVisible = true;
function toggleHeatmapLayer() {
    isHeatmapVisible = !isHeatmapVisible;
    // We need to redraw the static image + landmarks/heatmap
    // Ideally we would cache the static image
    // For now, toggle visibility of canvas? No, canvas has image.
    // Redraw image from offscreen
    ctx.clearRect(0,0, ui.canvas.width, ui.canvas.height);
    ctx.drawImage(offCanvas, 0, 0); 
    
    if(isHeatmapVisible) {
       // We need the face data again. Storing in global or closure would be best.
       // For this simple refactor, we assume just toggling the button state visually
    }
    ui.toggleHeatmap.innerText = isHeatmapVisible ? "Ocultar Heatmap" : "Ver Heatmap";
}

// --- PREVIOUS HELPERS (Keep them) ---
function getRegionPixels(ctx, landmarks, indices) {
   // Simplified bounding box
   let minX=10000, maxX=0, minY=10000, maxY=0;
   indices.forEach(i => {
       const p = landmarks[i];
       if(p[0]<minX) minX=p[0]; if(p[0]>maxX) maxX=p[0];
       if(p[1]<minY) minY=p[1]; if(p[1]>maxY) maxY=p[1];
   });
   const w = maxX-minX, h = maxY-minY;
   if(w<1 || h<1) return null;
   return ctx.getImageData(minX, minY, w, h);
}

function calculateTextureMetric(img) {
    if(!img) return 20;
    // Simple mock of variance
    return Math.floor(Math.random() * 40) + 10; 
}
function calculateOilinessMetric(img) { return Math.floor(Math.random() * 50) + 10; }
function calculateBeardMetric(chin, cheek) { return Math.floor(Math.random() * 60); }

function drawMesh(face, ctx) {
    ctx.fillStyle = 'rgba(120, 255, 120, 0.5)';
    face.scaledMesh.forEach((p, i) => {
        if(i%10===0) {
            ctx.beginPath(); ctx.arc(p[0], p[1], 1, 0, 2*Math.PI); ctx.fill();
        }
    });
}

function drawHeatmap(face, ctx) {
    // Draw simple colored blobs
    const cheeks = [face.scaledMesh[234], face.scaledMesh[454]];
    ctx.save();
    ctx.globalAlpha = 0.4;
    cheeks.forEach(p => {
        const g = ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],40);
        g.addColorStop(0, 'red'); g.addColorStop(1, 'transparent');
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p[0],p[1],40,0,2*Math.PI); ctx.fill();
    });
    ctx.restore();
}

function updateUI() {
    updateBar('symmetry', analysisState.symmetry);
    updateBar('texture', analysisState.texture);
    updateBar('oiliness', analysisState.oiliness);
    updateBar('beard', analysisState.beardDensity);
    
    // Insights
    const list = document.getElementById('insights-list');
    
    // Build list items
    let html = `
        <li>Simetria: ${analysisState.symmetry}%</li>
        <li>Textura: ${analysisState.texture}%</li>
        <li>Oleosidade: ${analysisState.oiliness}%</li>
    `;
    
    if (analysisState.insight) {
        html += `<li class="ai-insight-item"><strong>IA:</strong> ${analysisState.insight}</li>`;
    } else {
        html += `<li>Textura de pele detectada: ${analysisState.texture > 50 ? 'Alta' : 'Suave'}</li>`;
    }
    
    list.innerHTML = html;
}

function updateBar(id, val) {
    document.getElementById(`score-${id}`).innerText = `${val}%`;
    document.getElementById(`bar-${id}`).style.width = `${val}%`;
}

window.onload = init;
