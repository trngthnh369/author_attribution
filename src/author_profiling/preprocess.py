import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

def init_nlp_tools():
    """
    Hàm khởi tạo và đảm bảo tài nguyên NLTK đầy đủ.
    
    Returns:
        tuple: (stemmer, lemmatizer)
    """
    # Tải các tài nguyên cần thiết nếu chưa có
    nltk.download('punkt')
    nltk.download('wordnet')
    
    # Khởi tạo công cụ
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    return stemmer, lemmatizer

def stem_lema(text):
    """
    Tiền xử lý văn bản:
    - Chuyển về chữ thường
    - Tokenize (tách từ)
    - Loại bỏ dấu câu
    - Lemmatization
    - Stemming

    Args:
        text (str): Văn bản đầu vào

    Returns:
        list: Danh sách từ đã được xử lý
    """
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = []
    stemmer, lemmatizer = init_nlp_tools()

    for token in tokens:
        # Bỏ qua token là dấu câu
        if token in string.punctuation:
            continue
        
        lemma = lemmatizer.lemmatize(token)
        stem = stemmer.stem(lemma)
        processed_tokens.append(stem)

    return processed_tokens

def test():
    sample_text = "Running runners easily ran towards better universities."
    
    print("Văn bản gốc:")
    print(sample_text)
        
    processed = stem_lema(sample_text)
        
    print("\nKết quả sau khi Lemmatization + Stemming:")
    print(processed)