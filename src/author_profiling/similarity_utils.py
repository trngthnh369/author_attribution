# from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_sklearn(profile1, profile2):
    """
    Hàm tính Cosine Similarity giữa hai profile n-gram dùng sklearn.
    
    Args:
        profile1 (Counter): Bộ đếm n-gram của văn bản 1
        profile2 (Counter): Bộ đếm n-gram của văn bản 2
        
    Returns:
        float: Giá trị Cosine Similarity (0 đến 1)
    """
    
    all_ngrams = set(profile1.keys()).union(set(profile2.keys()))
    
    vec1 = []
    vec2 = []
    
    for ngram in all_ngrams:
        vec1.append(profile1.get(ngram, 0))
        vec2.append(profile2.get(ngram, 0))
        
    # Biến đổi thành mảng 2D theo đúng yêu cầu của sklearn
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    
    return similarity