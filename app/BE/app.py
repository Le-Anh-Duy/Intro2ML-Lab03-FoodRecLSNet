from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import io
import logging
import base64
import urllib.request
import numpy as np
import cv2
from model_config import MODEL_CONFIGS

# --- MMDET IMPORTS ---
try:
    from mmcv import Config
    from mmdet.apis import init_detector, inference_detector
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("Warning: mmdet not installed. Detection models will not work.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- PATH SETUP ---
# Add lsnet/detection to system path to allow imports of custom models
current_dir = os.path.dirname(os.path.abspath(__file__))
lsnet_detection_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'lsnet', 'detection'))

if lsnet_detection_dir not in sys.path:
    sys.path.insert(0, lsnet_detection_dir)
    logger.info(f"Added {lsnet_detection_dir} to sys.path")

# Force reload of 'model' package to pick up detection models
# This is necessary because 'model' package was already loaded by model_config.py (pointing to classification models)
if 'model' in sys.modules:
    # We need to remove 'model' and all its submodules to ensure clean import
    modules_to_remove = [m for m in sys.modules if m == 'model' or m.startswith('model.')]
    for m in modules_to_remove:
        del sys.modules[m]
    logger.info(f"Removed {len(modules_to_remove)} modules from sys.modules to allow detection model import")

# Import custom model modules for detection
try:
    import model.lsnet
    import model.lsnet_fpn
    
    # Verify registration
    if MMDET_AVAILABLE:
        from mmdet.models.builder import BACKBONES
        if 'lsnet_t' in BACKBONES:
            logger.info("lsnet_t successfully registered in BACKBONES")
        else:
            logger.error("lsnet_t NOT registered in BACKBONES after import")
        
    logger.info("Successfully imported custom model modules for detection")
except ImportError as e:
    logger.warning(f"Failed to import custom model modules for detection: {e}")

# --- GLOBAL VARIABLES ---
LOADED_MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def download_file_if_missing(url, path):
    """Helper function to download file if it doesn't exist."""
    if not url: return
    if not os.path.exists(path):
        logger.info(f"File not found: {path}. Downloading from {url}...")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
            logger.info(f"Downloaded successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to download {path}: {e}")

def load_models():
    """Load all configured models into memory."""
    logger.info(f"Loading models on {DEVICE}...")
    
    for config in MODEL_CONFIGS:
        model_id = config["id"]
        model_type = config.get("type", "classification")
        weights_path = config["weights_path"]
        classes_path = config["classes_path"]
        
        # --- 1. DOWNLOAD FILES IF MISSING ---
        download_file_if_missing(config.get("classes_url"), classes_path)
        download_file_if_missing(config.get("weights_url"), weights_path)

        # --- 2. LOAD CLASSES ---
        classes = []
        if os.path.exists(classes_path):
            try:
                with open(classes_path, "r", encoding="utf-8") as f:
                    classes = [s.strip() for s in f.readlines()]
                logger.info(f"Loaded {len(classes)} classes from {classes_path}")
            except Exception as e:
                logger.error(f"Error loading classes for {model_id}: {e}")
        else:
            logger.warning(f"Classes file still not found: {classes_path}")

        # --- 3. LOAD MODEL ---
        if not os.path.exists(weights_path):
            logger.warning(f"Weights file not found: {weights_path}. Skipping {model_id}")
            continue

        try:
            if model_type == "classification":
                arch_fn = config.get("arch_fn")
                if arch_fn is None:
                    logger.error(f"Architecture function for {model_id} is missing.")
                    continue
                    
                logger.info(f"Loading classification model {model_id}...")
                model = arch_fn(num_classes=config["num_classes"], pretrained=False)
                
                checkpoint = torch.load(weights_path, map_location=DEVICE)
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                
            elif model_type == "detection":
                if not MMDET_AVAILABLE:
                    logger.error(f"MMDet not available. Skipping detection model {model_id}")
                    continue
                    
                config_path = config.get("config_path")
                if not os.path.exists(config_path):
                    logger.error(f"Config file not found: {config_path}")
                    continue
                    
                logger.info(f"Loading detection model {model_id}...")
                
                try:
                    # --- FIX LỖI PRETRAIN PATH (CẬP NHẬT) ---
                    cfg = Config.fromfile(config_path)
                    
                    # Can thiệp trực tiếp vào dictionary của config
                    if 'model' in cfg:
                        # Xử lý backbone
                        if 'backbone' in cfg.model:
                            backbone = cfg.model['backbone']
                            
                            # 1. Xóa init_cfg (cách mới của mmdet)
                            if 'init_cfg' in backbone:
                                logger.info(f"Removing init_cfg from backbone: {backbone['init_cfg']}")
                                backbone['init_cfg'] = None
                            
                            # 2. Xóa pretrained (cách cũ hoặc custom code)
                            if 'pretrained' in backbone:
                                logger.info(f"Removing pretrained from backbone: {backbone['pretrained']}")
                                backbone['pretrained'] = None
                        
                        # Xử lý pretrained ở cấp model (nếu có)
                        if 'pretrained' in cfg.model:
                            cfg.model['pretrained'] = None

                    # Khởi tạo model với config đã sửa
                    model = init_detector(cfg, weights_path, device=str(DEVICE))
                    
                    # --- CẬP NHẬT CLASS NAMES ---
                    try:
                        model.CLASSES = tuple(config.classes)
                        logger.info(f"Loaded {len(config.classes)} custom classes.")
                    except Exception as e:
                        logger.warning(f"Could not load classes file: {e}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize detector: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            LOADED_MODELS[model_id] = {
                "model": model,
                "config": config,
                "classes": classes,
                "type": model_type
            }
            logger.info(f"Successfully loaded model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            import traceback
            traceback.print_exc()

# Initialize models
load_models()

@app.route('/models', methods=['GET'])
def get_models():
    """Return list of available models with details."""
    models_list = []
    for model_id, info in LOADED_MODELS.items():
        models_list.append({
            'id': model_id,
            'name': info['config']['name'],
            'type': info['type']
        })
    
    return jsonify({
        'success': True,
        'models': models_list
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not LOADED_MODELS:
        return jsonify({
            'success': False,
            'message': 'No models loaded available for prediction.'
        }), 503

    try:
        data = request.json
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400

        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get model_id
        requested_model_id = data.get('model_id')
        if requested_model_id and requested_model_id in LOADED_MODELS:
            model_id = requested_model_id
        else:
            model_id = list(LOADED_MODELS.keys())[0]
            
        model_info = LOADED_MODELS[model_id]
        model = model_info["model"]
        classes = model_info["classes"]
        model_type = model_info["type"]
        
        result_image_b64 = None
        predictions = []

        if model_type == "classification":
            # Preprocess
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            for i in range(top5_prob.size(0)):
                idx = top5_catid[i].item()
                score = top5_prob[i].item()
                label = classes[idx] if classes and idx < len(classes) else str(idx)
                predictions.append({
                    'name': label,
                    'confidence': float(score)
                })
                
        elif model_type == "detection":
            # Convert PIL to OpenCV (RGB -> BGR)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Inference
            result = inference_detector(model, image_cv)
            
            # Process results
            if isinstance(result, tuple):
                bbox_result, segm_result = result
            else:
                bbox_result, segm_result = result, None
                
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            
            # Filter & NMS
            score_thr = 0.3
            scores = bboxes[:, -1]
            inds = scores > score_thr

            # remove same labels (get only highest score for each label)
            unique_labels = {}

            for label, bbox in zip(labels[inds], bboxes[inds]):
                score = bbox[-1]
                if label not in unique_labels or score > unique_labels[label][-1]:
                    unique_labels[label] = bbox

            bboxes = np.array(list(unique_labels.values()))
            labels = np.array(list(unique_labels.keys()))

            # bboxes = bboxes[inds, :]
            # labels = labels[inds]
            
            if len(bboxes) > 0:
                # NMS
                boxes_xywh = []
                scores_list = []
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox
                    boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                    scores_list.append(float(score))
                
                nms_indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_list, score_thr, 0.3)
                
                if len(nms_indices) > 0:
                    nms_indices = nms_indices.flatten()
                    bboxes = bboxes[nms_indices]
                    labels = labels[nms_indices]
                    
                    # Draw boxes
                    for bbox, label in zip(bboxes, labels):
                        x1, y1, x2, y2, score = bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get label name
                        if hasattr(model, 'CLASSES') and label < len(model.CLASSES):
                            label_text = f"{model.CLASSES[label]}"
                        else:
                            label_text = f"Class {label}"
                            
                        # Add to predictions list
                        predictions.append({
                            'name': label_text,
                            'confidence': float(score),
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # Draw on image
                        color = (0, 255, 0)
                        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
                        
                        text = f"{label_text} {score:.2f}"
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(image_cv, (x1, y1 - 25), (x1 + w, y1), color, -1)
                        cv2.putText(image_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Encode result image
            _, buffer = cv2.imencode('.jpg', image_cv)
            result_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Sort predictions by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'success': True,
            # get at most 5 predictions
            'predictions': predictions,
            'model_used': model_id,
            'result_image': result_image_b64
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing image'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Backend is running',
        'loaded_models': list(LOADED_MODELS.keys())
    }), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Vietnamese Food Recognition API',
        'endpoints': {
            '/predict': 'POST - Predict food from image',
            '/health': 'GET - Check server status',
            '/models': 'GET - List available models'
        }
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Vietnamese Food Recognition Backend...")
    print(f"📍 Server running at: http://localhost:5000")
    print(f"📱 Loaded models: {list(LOADED_MODELS.keys())}")
    app.run(debug=True, host='0.0.0.0', port=5000)
