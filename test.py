import nltk

# Cài đặt và tải tài nguyên
nltk.download('punkt', quiet=True)

# Xác nhận các đường dẫn tải dữ liệu
print("NLTK data paths:")
for path in nltk.data.path:
    print(path)

# Kiểm tra tài nguyên đã tải
try:
    nltk.data.find('tokenizers/punkt')
    print("Resource punkt is available.")
except LookupError:
    print("Resource punkt not found.")
