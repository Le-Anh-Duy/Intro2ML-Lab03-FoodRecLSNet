import torch

# Đường dẫn đến 2 file của bạn
path_ok = "D:\Coding\School\Y3-K1\Intro2ML\Lab 3 - DeepLearning\src\\app\BE\pretrained\lsnet_vietnamesefood_best.pth"
path_error = "D:\Coding\School\Y3-K1\Intro2ML\Lab 3 - DeepLearning\src\\app\BE\pretrained\lsnet_t_finetuned.pth"

def inspect_file(path):
    print(f"\n--- Đang kiểm tra: {path} ---")
    try:
        # Load về CPU để tránh lỗi CUDA nếu máy không có GPU
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Kiểm tra kiểu dữ liệu
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            # Nếu là dict, kiểm tra xem nó là state_dict hay là checkpoint chứa cả optimizer, epoch...
            checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint

            keys = list(checkpoint.keys())
            print(f"Các key cấp 1: {keys[:5]} ... (Tổng: {len(keys)})")
            
            # Nếu key đầu tiên trông giống tên layer (vd: 'conv1.weight') thì đây là state_dict
            if 'state_dict' in keys:
                print(">> Đây là Checkpoint tổng hợp (chứa state_dict, optimizer, epoch...)")
                return checkpoint['state_dict']
            else:
                print(">> Đây có vẻ là state_dict thuần túy.")
                return checkpoint
        else:
            print(">> Đây là Full Model (được lưu bằng torch.save(model)).")
            return checkpoint.state_dict() # Lấy state_dict ra để so sánh
            
    except Exception as e:
        print(f"LỖI KHI ĐỌC FILE: {e}")
        return None

# Lấy dữ liệu 2 file
state_dict_1 = inspect_file(path_ok)
state_dict_2 = inspect_file(path_error)

# So sánh chi tiết nếu cả 2 đều đọc được
if state_dict_1 and state_dict_2:
    keys1 = set(state_dict_1.keys())
    keys2 = set(state_dict_2.keys())
    
    print("\n--- KẾT QUẢ SO SÁNH ---")
    if keys1 == keys2:
        print("✅ Tên các Layer khớp nhau hoàn toàn.")
        # Kiểm tra Shape (kích thước)
        for key in keys1:
            if state_dict_1[key].shape != state_dict_2[key].shape:
                print(f"❌ Sai khác kích thước tại layer: {key}")
                print(f"   File OK: {state_dict_1[key].shape}")
                print(f"   File Lỗi: {state_dict_2[key].shape}")
    else:
        print("❌ Tên các Layer KHÔNG khớp nhau.")
        diff_1 = keys1 - keys2
        diff_2 = keys2 - keys1
        if diff_1: print(f"Key chỉ có ở file OK (ví dụ): {list(diff_1)[:3]}")
        if diff_2: print(f"Key chỉ có ở file Lỗi (ví dụ): {list(diff_2)[:3]}")