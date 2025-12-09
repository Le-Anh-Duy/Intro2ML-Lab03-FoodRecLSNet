from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Danh sách món ăn Việt Nam mẫu
VIETNAMESE_FOODS = [
    "Phở bò", "Phở gà", "Bún chả", "Bún bò Huế", "Bánh mì",
    "Cơm tấm", "Bánh xèo", "Gỏi cuốn", "Nem rán", "Chả giò",
    "Cao lầu", "Mì Quảng", "Hủ tiếu", "Bánh cuốn", "Bún riêu",
    "Cháo lòng", "Bánh bao", "Xôi", "Chè", "Bánh flan",
    "Bún đậu mắm tôm", "Bánh khọt", "Bánh bèo", "Nem nướng", "Bún thịt nướng"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Lấy dữ liệu ảnh (base64)
        image_data = data.get('image', '')
        selection = data.get('selection', None)
        
        # Xử lý ảnh (chỉ để validate, không dùng cho model)
        if image_data:
            # Loại bỏ header "data:image/jpeg;base64,"
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            print(f"Received image size: {image.size}")
            if selection:
                print(f"Selection area: {selection}")
        
        # Tạo kết quả dummy với xác suất ngẫu nhiên
        num_results = random.randint(3, 5)
        selected_foods = random.sample(VIETNAMESE_FOODS, num_results)
        
        # Tạo confidence scores giảm dần
        confidences = sorted([random.uniform(0.5, 0.98) for _ in range(num_results)], reverse=True)
        
        predictions = [
            {
                'name': food,
                'confidence': confidence
            }
            for food, confidence in zip(selected_foods, confidences)
        ]
        
        response = {
            'success': True,
            'predictions': predictions,
            'message': 'Đây là kết quả dummy. Model thực sẽ được tích hợp sau.'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Đã xảy ra lỗi khi xử lý ảnh'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Backend đang hoạt động'
    }), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Vietnamese Food Recognition API',
        'endpoints': {
            '/predict': 'POST - Nhận diện món ăn',
            '/health': 'GET - Kiểm tra trạng thái server'
        }
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Vietnamese Food Recognition Backend...")
    print("📍 Server running at: http://localhost:5000")
    print("🔍 Endpoints:")
    print("   - POST /predict - Nhận diện món ăn")
    print("   - GET /health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5000)
