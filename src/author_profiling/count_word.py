from nltk.util import ngrams
from collections import Counter
from .preprocess import stem_lema

def count_word(text_tokens, n=2):
    """
    Hàm tính n-gram profile của văn bản sau tiền xử lý.
    
    Args:
        text (str): Văn bản đầu vào
        n (int): Giá trị n của n-gram (mặc định là bigram)
        
    Returns:
        Counter: Bộ đếm tần suất các n-gram
    """
    n_gram_list = list(ngrams(text_tokens, n))
    profile = Counter(n_gram_list)
    
    return profile