// DOM Elements
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadForm = document.getElementById('uploadForm');
const detectBtn = document.getElementById('detectBtn');
const startWebcamBtn = document.getElementById('startWebcam');
const stopWebcamBtn = document.getElementById('stopWebcam');
const captureBtn = document.getElementById('captureBtn');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const resultDisplay = document.getElementById('resultDisplay');
const loading = document.getElementById('loading');
const predictionResult = document.getElementById('predictionResult');
const predictedLetter = document.getElementById('predictedLetter');
const confidenceValue = document.getElementById('confidenceValue');
const resultImage = document.getElementById('resultImage');
const letterDescription = document.getElementById('letterDescription');
const alphabetGrid = document.getElementById('alphabetGrid');

let stream = null;

// ASL Alphabet data (you can expand this)
const aslAlphabet = {
    'A': { description: 'Closed fist with thumb alongside.' },
    'B': { description: 'Flat hand, fingers together, thumb across palm.' },
    'C': { description: 'Curved hand in a C shape.' },
    'D': { description: 'Pointing up with index finger, others in fist.' },
    'E': { description: 'Fingers curled in, thumb across fingers.' },
    // Add more letters as needed
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeAlphabetGrid();
    setupEventListeners();
});

// Setup alphabet reference grid
function initializeAlphabetGrid() {
    alphabetGrid.innerHTML = '';
    for (let letter = 65; letter <= 90; letter++) {
        const char = String.fromCharCode(letter);
        const div = document.createElement('div');
        div.className = 'alphabet-letter';
        div.textContent = char;
        div.title = `ASL letter ${char}`;
        div.addEventListener('click', () => showLetterInfo(char));
        alphabetGrid.appendChild(div);
    }
}

// Show letter information
function showLetterInfo(letter) {
    const info = aslAlphabet[letter] || { description: 'Information not available.' };
    alert(`ASL Letter ${letter}:\n${info.description}`);
}

// Event Listeners
function setupEventListeners() {
    // File upload
    dropArea.addEventListener('click', () => fileInput.click());
    
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = '#1e3c72';
        dropArea.style.background = '#eef2ff';
    });
    
    dropArea.addEventListener('dragleave', () => {
        dropArea.style.borderColor = '#2a5298';
        dropArea.style.background = '#f8faff';
    });
    
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = '#2a5298';
        dropArea.style.background = '#f8faff';
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            displayFileName();
        }
    });
    
    fileInput.addEventListener('change', displayFileName);
    
    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select an image first!');
            return;
        }
        
        showLoading();
        
        // Submit form to Flask
        const formData = new FormData(this);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            displayResult(data);
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            alert('Error detecting sign. Please try again.');
        });
    });
    
    // Webcam controls
    startWebcamBtn.addEventListener('click', startWebcam);
    stopWebcamBtn.addEventListener('click', stopWebcam);
    captureBtn.addEventListener('click', captureImage);
}

// Display selected file name
function displayFileName() {
    if (fileInput.files.length) {
        fileName.textContent = `Selected: ${fileInput.files[0].name}`;
        fileName.style.display = 'block';
    } else {
        fileName.style.display = 'none';
    }
}

// Webcam functions
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        webcam.srcObject = stream;
        
        startWebcamBtn.disabled = true;
        stopWebcamBtn.disabled = false;
        captureBtn.disabled = false;
        
        startWebcamBtn.classList.add('hidden');
        stopWebcamBtn.classList.remove('hidden');
    } catch (err) {
        console.error('Error accessing webcam:', err);
        alert('Could not access webcam. Please check permissions.');
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        
        startWebcamBtn.disabled = false;
        stopWebcamBtn.disabled = true;
        captureBtn.disabled = true;
        
        startWebcamBtn.classList.remove('hidden');
        stopWebcamBtn.classList.add('hidden');
    }
}

function captureImage() {
    if (!stream) return;
    
    const context = canvas.getContext('2d');
    context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to blob and upload
    canvas.toBlob(function(blob) {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        
        // Create form and submit
        const formData = new FormData();
        formData.append('file', file);
        
        showLoading();
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            displayResult(data);
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            alert('Error detecting sign. Please try again.');
        });
    }, 'image/jpeg');
}

// Loading states
function showLoading() {
    loading.style.display = 'block';
    resultDisplay.style.opacity = '0.5';
    detectBtn.disabled = true;
    detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
}

function hideLoading() {
    loading.style.display = 'none';
    resultDisplay.style.opacity = '1';
    detectBtn.disabled = false;
    detectBtn.innerHTML = '<i class="fas fa-search"></i> Detect Sign';
}

// Display results from Flask
function displayResult(data) {
    predictionResult.style.display = 'block';
    document.querySelector('.placeholder-result').style.display = 'none';
    
    // Update with actual data from Flask
    predictedLetter.textContent = data.letter || 'A';
    confidenceValue.textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A';
    
    if (data.image) {
        resultImage.src = data.image;
        resultImage.style.display = 'block';
    }
    
    // Get description for the letter
    const letter = data.letter || 'A';
    const description = aslAlphabet[letter]?.description || 'No description available.';
    letterDescription.textContent = description;
}