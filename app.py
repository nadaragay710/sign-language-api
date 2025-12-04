import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load files
print("🚀 Loading sign language model...")

# Try to load model, but continue even if it fails
try:
    model = tf.keras.models.load_model('sign_language_model_final.h5')
    print("✅ Model loaded successfully")
except:
    print("⚠️  Could not load model file")
    model = None

try:
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    print(f"📊 Classes: {list(class_mapping.keys())}")
except:
    print("⚠️  Could not load class mapping")
    class_mapping = {"Hello": 0, "Thank You": 1}  # Default

def predict_sign(image):
    """Simple prediction function"""
    if model is None:
        return "Model not loaded. Please check if model file exists."
    
    try:
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Simple preprocessing
        image = image.resize((64, 64)).convert('L')  # Resize to 64x64 grayscale
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Reshape for model
        img_array = img_array.reshape(1, 64, 64, 1)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Get class name
        class_names = list(class_mapping.keys())
        class_name = class_names[class_idx]
        
        return f"**Prediction:** {class_name}\n**Confidence:** {confidence:.1%}"
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create the web interface
with gr.Blocks() as demo:
    gr.Markdown("# ✋ Sign Language to Text Translator")
    gr.Markdown("Upload an image of a sign language gesture to get the text translation")
    
    with gr.Row():
        with gr.Column():
            # Webcam or file upload
            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                label="Upload or Capture Sign Language Image"
            )
            
            # Predict button
            predict_btn = gr.Button("Translate", variant="primary", size="lg")
            
            # Example text
            gr.Markdown("**Examples you can recognize:**")
            gr.Markdown("- Hello\n- Thank You\n- Yes\n- No\n- I Love You")
        
        with gr.Column():
            # Output
            output = gr.Markdown("## Prediction will appear here")
    
    # Connect button to function
    predict_btn.click(
        fn=predict_sign,
        inputs=image_input,
        outputs=output
    )
    
    # Auto-predict when image is uploaded
    image_input.change(
        fn=predict_sign,
        inputs=image_input,
        outputs=output
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("Built with TensorFlow and Gradio | Model: Sign Language Recognition")

# Launch
if __name__ == "__main__":
    demo.launch()