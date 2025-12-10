import os
import sys
import torch
import logging
import numpy as np
import cv2
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
lsnet_detection_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'lsnet', 'detection'))

if lsnet_detection_dir not in sys.path:
    sys.path.insert(0, lsnet_detection_dir)
    logger.info(f"Added {lsnet_detection_dir} to sys.path")

# Import custom model modules
try:
    import model.lsnet
    import model.lsnet_fpn
    logger.info("Successfully imported custom model modules")
except ImportError as e:
    logger.error(f"Failed to import custom model modules: {e}")
    sys.exit(1)

def visualize_result(image_path, bboxes, labels, class_names, score_thr=0.3, out_file=None):
    """
    Vẽ bounding boxes và nhãn lên ảnh, sau đó hiển thị và lưu file.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not read image {image_path}")
        return

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = bbox
        if score < score_thr:
            continue
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Màu xanh lá
        color = (0, 255, 0)
        
        # Vẽ khung chữ nhật
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Tạo nội dung nhãn
        if class_names and label < len(class_names):
            label_text = f"{class_names[label]} {score:.2f}"
        else:
            label_text = f"Class {label} {score:.2f}"
            
        # Vẽ nền cho chữ để dễ đọc
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        
        # Vẽ chữ màu đen
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if out_file:
        cv2.imwrite(out_file, img)
        logger.info(f"Saved visualization to {out_file}")

    # Hiển thị ảnh (nhấn phím bất kỳ để đóng)
    cv2.imshow('Detection Result', img)
    logger.info("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_image(image_path, config_path, checkpoint_path, classes_path=None, device='cuda:0'):
    """
    Run object detection on an image.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return

    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Initializing detector with config: {config_path}")
    
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
        model = init_detector(cfg, checkpoint_path, device=device)
        
        # --- CẬP NHẬT CLASS NAMES ---
        if classes_path and os.path.exists(classes_path):
            try:
                with open(classes_path, 'r', encoding='utf-8') as f:
                    custom_classes = [line.strip() for line in f.readlines()]
                model.CLASSES = tuple(custom_classes)
                logger.info(f"Loaded {len(custom_classes)} custom classes.")
            except Exception as e:
                logger.warning(f"Could not load classes file: {e}")
        
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info(f"Running inference on: {image_path}")
    
    try:
        result = inference_detector(model, image_path)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

    # Show results
    out_file = os.path.join(current_dir, 'result.jpg')
    
    # Print text results
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
    
    # Filter by score threshold
    score_thr = 0.3
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    # --- APPLY NMS (Non-Maximum Suppression) ---
    # Loại bỏ các box chồng chéo nhau
    if len(bboxes) > 0:
        # Chuẩn bị dữ liệu cho cv2.dnn.NMSBoxes: [x, y, w, h]
        boxes_xywh = []
        scores_list = []
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores_list.append(float(score))
        
        # nms_threshold=0.5: Nếu IoU > 0.5 thì loại bỏ box có điểm thấp hơn
        nms_indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_list, score_thr, 0.5)
        
        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            bboxes = bboxes[nms_indices]
            labels = labels[nms_indices]
        else:
            bboxes = np.array([])
            labels = np.array([])
    
    print("\n--- DETECTION RESULTS ---")
    if len(bboxes) > 0:
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            if hasattr(model, 'CLASSES') and label < len(model.CLASSES):
                class_name = model.CLASSES[label]
            else:
                class_name = str(label)
            score = bbox[-1]
            print(f"Object {i+1}: {class_name} ({score*100:.2f}%) - Box: {bbox[:4]}")
    else:
        print("No objects detected above threshold.")

    # --- VISUALIZE ---
    visualize_result(
        image_path, 
        bboxes, 
        labels, 
        model.CLASSES if hasattr(model, 'CLASSES') else None,
        score_thr=score_thr,
        out_file=out_file
    )

if __name__ == "__main__":
    CONFIG_FILE = os.path.join(lsnet_detection_dir, "configs", "retinanet_lsnet_b_fpn_1x_coco.py")
    CHECKPOINT_FILE = os.path.join(current_dir, "pretrained", "retinanet_lsnet_b_finetuned.pth")
    IMAGE_FILE = os.path.join(current_dir, "dumps/comtam.jpg") 
    CLASSES_FILE = os.path.join(current_dir, "pretrained", "vietnamese_food_classes.txt")
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    detect_image(IMAGE_FILE, CONFIG_FILE, CHECKPOINT_FILE, classes_path=CLASSES_FILE, device=DEVICE)
