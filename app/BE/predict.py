import torch
from torchvision import transforms
from PIL import Image
import urllib.request

# --- 1. IMPORT MODEL CỤC BỘ (QUAN TRỌNG) ---
# Đảm bảo bạn import đúng đường dẫn file lsnet.py chứa class SKA đã sửa
# Nếu file lsnet.py nằm trong thư mục models, dùng dòng dưới:

import sys
import os

# Get the absolute path to the directory containing the file you want to import
# Example: If 'my_module.py' is in '../other_directory' relative to the current script
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lsnet'))

# Add the directory to sys.path
sys.path.insert(0, module_dir) # Use insert(0, ...) to prioritize this path


from model.lsnet import *
# Nếu lsnet.py nằm ngay bên ngoài, dùng: from lsnet import lsnet_t

def download_weights(url, save_path):
    try:
        print(f"Đang tải weights từ {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Tải xong và lưu tại {save_path}")
    except Exception as e:
        print(f"LỖI khi tải weights: {e}")

def predict_image(image_path, model_path):
    device = torch.device("cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # --- 2. KHỞI TẠO MODEL ---
    print("Đang khởi tạo LSNet Tiny...")
    model = lsnet_t_distill(num_classes=30, pretrained=False) # False vì ta load thủ công bên dưới
    
    # --- 3. LOAD WEIGHTS ---
    print(f"Đang load weights từ {model_path}...")
    try:
        # map_location='cpu' là BẮT BUỘC
        checkpoint = torch.load(model_path, map_location=device)
        
        # Xử lý cấu trúc file weight (đôi khi nó nằm trong key 'model')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Chế độ suy luận (tắt Dropout/BatchNorm)
        print("Load model thành công!")
    except Exception as e:
        print(f"LỖI load weight: {e}")
        return

    # --- 4. XỬ LÝ ẢNH (PREPROCESSING) ---
    # Chuẩn hóa theo chuẩn ImageNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    try:
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # Tạo batch size = 1 [1, 3, 224, 224]
        input_batch = input_batch.to(device)
    except FileNotFoundError:
        print(f"Không tìm thấy ảnh: {image_path}")
        return

    # --- 5. TẢI NHÃN (LABELS) ---
    # Tải danh sách tên 1000 loại vật thể của ImageNet
    url = "https://huggingface.co/MatchaMacchiato/LSNet_VietnameseFood/resolve/main/vietnamese_food_classes.txt?download=true"
    try:
        filename = "vietnamese_food_classes.txt"
        urllib.request.urlretrieve(url, filename)
        with open(filename, "r") as f:
            categories = [s.strip() for s in f.readlines()]
    except:
        categories = None
        print("Không tải được danh sách tên, sẽ hiển thị ID số.")

    # --- 6. DỰ ĐOÁN (INFERENCE) ---
    print("Đang phân tích ảnh...")
    with torch.no_grad():
        output = model(input_batch)
    
    # Tính xác suất (Softmax)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Lấy Top 5 kết quả cao nhất
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        score = top5_prob[i].item()
        label = categories[idx] if categories else str(idx)
        print(f"Hạng {i+1}: {label} ({score*100:.2f}%)")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Thay đổi tên file ảnh của bạn ở đây
    IMAGE_FILE = "comtam.jpg" 
    # Thay đổi đường dẫn file weight tải về ở đây
    WEIGHT_FILE = "lsnet_vietnamesefood_best.pth"   
    
    predict_image(IMAGE_FILE, WEIGHT_FILE)