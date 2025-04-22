# Sử dụng image PyTorch CPU base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime



# Cài đặt các thư viện bổ sung (nếu cần)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Chuyển thư mục làm việc sang /app
WORKDIR /app

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Cài đặt các dependencies từ requirements.txt (không cần cài PyTorch nữa)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Lệnh chạy khi container khởi động (có thể thay đổi script tuỳ ý)
CMD ["python", "train.py"]
