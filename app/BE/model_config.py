import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PATH SETUP ---
# Add lsnet to system path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
lsnet_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'lsnet'))

if lsnet_dir not in sys.path:
    sys.path.insert(0, lsnet_dir)
    logger.info(f"Added {lsnet_dir} to sys.path")

# Import model architecture
try:
    from model.lsnet import lsnet_t_distill
    from model.lsnet import lsnet_t
    from model.lsnet import lsnet_b
    from model.lsnet import lsnet_s
    logger.info("Successfully imported lsnet_t_distill")
except ImportError as e:
    logger.error(f"Failed to import lsnet: {e}")
    lsnet_t_distill = None
    lsnet_t = None
    lsnet_s = None
    lsnet_b = None

# --- CONFIGURATION ---
MODEL_CONFIGS = [
    {
        "id": "lsnet_tiny_distill",
        "type": "classification",
        "name": "LSNet Tiny Distill (30 classes Vietnamese Food)",
        "weights_path": os.path.join(current_dir, "pretrained", "lsnet_vietnamesefood_best.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes.txt"),
        "weights_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/lsnet_vietnamesefood_best.pth?download=true",
        "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
        "num_classes": 30,
        "arch_fn": lsnet_t_distill
    },    
    {
        "id": "lsnet_small",
        "type": "classification",
        "name": "LSNet small (Vietnamese Food)",
        "weights_path": os.path.join(current_dir, "pretrained", "lsnet_s_finetuned.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes_103.txt"),
        # "weights_url": "https://www.kaggle.com/models/meowluvmatcha/finetuned-resnet-mobilenet-lsnet",
        # "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
        "num_classes": 103,
        "arch_fn": lsnet_s
    },    
    {
        "id": "lsnet_tiny",
        "type": "classification",
        "name": "LSNet Tiny (Vietnamese Food)",
        "weights_path": os.path.join(current_dir, "pretrained", "lsnet_t_finetuned.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes_103.txt"),
        # "weights_url": "https://www.kaggle.com/models/meowluvmatcha/finetuned-resnet-mobilenet-lsnet",
        # "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
        "num_classes": 103,
        "arch_fn": lsnet_t
    },    
    {
        "id": "lsnet_base",
        "type": "classification",
        "name": "LSNet base (Vietnamese Food)",
        "weights_path": os.path.join(current_dir, "pretrained", "lsnet_b_finetuned.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes_103.txt"),
        # "weights_url": "https://www.kaggle.com/models/meowluvmatcha/finetuned-resnet-mobilenet-lsnet",
        # "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
        "num_classes": 103,
        "arch_fn": lsnet_b
    },
    {
        "id": "retinanet_lsnet_t",
        "type": "detection",
        "name": "LSNet Tiny Detection (RetinaNet)",
        "config_path": os.path.join(current_dir, "..", "..", "lsnet", "detection", "configs", "retinanet_lsnet_t_fpn_1x_food_coco.py"),
        "weights_path": os.path.join(current_dir, "pretrained", "retinanet_lsnet_t_finetuned.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes.txt"),
        "weights_url": "", # Add URL if available
        "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
    },
    {
        "id": "retinanet_lsnet_s",
        "type": "detection",
        "name": "LSNet Small Detection (RetinaNet)",
        "config_path": os.path.join(current_dir, "..", "..", "lsnet", "detection", "configs", "retinanet_lsnet_s_fpn_1x_food_coco.py"),
        "weights_path": os.path.join(current_dir, "pretrained", "retinanet_lsnet_s_finetuned.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes.txt"),
        "weights_url": "", # Add URL if available
        "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
    },
    {
        "id": "retinanet_lsnet_b",
        "type": "detection",
        "name": "LSNet Base Detection (RetinaNet)",
        "config_path": os.path.join(current_dir, "..", "..", "lsnet", "detection", "configs", "retinanet_lsnet_b_fpn_1x_food_coco.py"),
        "weights_path": os.path.join(current_dir, "pretrained", "retinanet_lsnet_b_finetuned.pth"),
        "classes_path": os.path.join(current_dir, "pretrained", "vietnamese_food_classes.txt"),
        "weights_url": "", # Add URL if available
        "classes_url": "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true",
    }
]
