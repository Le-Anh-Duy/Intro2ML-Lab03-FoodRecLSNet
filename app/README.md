# Ứng Dụng Nhận Diện Món Ăn Việt Nam

## 🎯 Tính năng
- ✅ Kéo thả hoặc upload ảnh món ăn
- ✅ Khoanh vùng món ăn cần nhận diện bằng chuột
- ✅ Gọi API backend để nhận diện
- ✅ Hiển thị kết quả với độ tin cậy

## 📁 Cấu trúc thư mục
```
app/
├── BE/
│   ├── app.py              # Backend Flask API
│   └── requirements.txt    # Dependencies
└── FE/
    └── index.html          # Frontend web app
```

## 🚀 Hướng dẫn chạy

### 1. Cài đặt Backend

```powershell
# Di chuyển vào thư mục BE
cd app\BE

# Tạo virtual environment (khuyến nghị)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy server
python app.py
```

Backend sẽ chạy tại: `http://localhost:5000`

### 2. Mở Frontend

Có 2 cách:

**Cách 1: Mở trực tiếp file HTML**
- Mở file `app/FE/index.html` bằng trình duyệt (Chrome, Firefox, Edge...)

**Cách 2: Dùng Live Server (khuyến nghị)**
- Cài extension "Live Server" trong VS Code
- Right-click vào `index.html` → chọn "Open with Live Server"
- Trang web sẽ mở tại `http://127.0.0.1:5500`

## 💡 Cách sử dụng

1. **Upload ảnh**: Kéo thả hoặc click để chọn file ảnh món ăn
2. **Khoanh vùng**: Nhấn và kéo chuột trên ảnh để tạo vùng chọn
3. **Nhận diện**: Click nút "Nhận Diện" để gửi request đến backend
4. **Xem kết quả**: Kết quả sẽ hiển thị với tên món ăn và độ tin cậy

## 🔧 API Endpoints

### POST /predict
Nhận diện món ăn từ ảnh

**Request body:**
```json
{
  "image": "base64_encoded_image_data",
  "selection": {
    "x": 100,
    "y": 100,
    "width": 300,
    "height": 300
  }
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "name": "Phở bò",
      "confidence": 0.95
    },
    {
      "name": "Bún chả",
      "confidence": 0.85
    }
  ],
  "message": "Đây là kết quả dummy..."
}
```

### GET /health
Kiểm tra trạng thái server

## 📝 Lưu ý

- **Backend hiện tại trả về kết quả dummy/random** để test UI
- Khi model đã được train xong, bạn chỉ cần:
  1. Load model trong `app.py`
  2. Thay thế logic trong hàm `predict()` để gọi model thực
  3. Xử lý vùng chọn (`selection`) nếu model hỗ trợ crop ảnh

## 🔮 Tích hợp Model thực

Khi đã có model, sửa file `app/BE/app.py`:

```python
import torch
from model.lsnet import LSNet  # Import model của bạn

# Load model
model = LSNet(...)
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # ... xử lý ảnh ...
    
    # Crop ảnh nếu có selection
    if selection:
        image = image.crop((
            selection['x'], 
            selection['y'],
            selection['x'] + selection['width'],
            selection['y'] + selection['height']
        ))
    
    # Gọi model
    predictions = model.predict(image)
    
    return jsonify({
        'success': True,
        'predictions': predictions
    })
```

## 🎨 Tùy chỉnh

- Sửa màu sắc: Thay đổi gradient trong CSS phần `body` và `.btn-primary`
- Thêm món ăn: Cập nhật list `VIETNAMESE_FOODS` trong `app.py`
- Kích thước canvas: Sửa `maxWidth`, `maxHeight` trong JavaScript

## ⚡ Troubleshooting

**Lỗi CORS:**
- Đảm bảo `flask-cors` đã được cài đặt
- Backend đã import và dùng `CORS(app)`

**Không kết nối được Backend:**
- Kiểm tra Backend đang chạy tại port 5000
- Kiểm tra URL trong frontend code (dòng `fetch('http://localhost:5000/predict')`)

**Ảnh không hiển thị:**
- Đảm bảo file ảnh có định dạng JPG, PNG hoặc JPEG
- Kiểm tra kích thước file không quá lớn
