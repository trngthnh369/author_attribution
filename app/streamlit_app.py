import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st

from author_profiling.preprocess import stem_lema
from author_profiling.count_word import count_word
from author_profiling.similarity_utils import cosine_similarity_sklearn

THRESHOLD = 0.8  # Ngưỡng phân biệt cùng tác giả

import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def compare_authors(test_text):
    """
    Hàm chính để so sánh tác giả của hai văn bản.
    - Đọc file alice.txt
    - Nhận văn bản cần kiểm tra từ tham số
    - Tiền xử lý và tính n-gram
    - Tính Cosine Similarity
    - Trả về giá trị similarity
    """
    # Đọc file alice.txt
    with open("data/alice.txt", "r", encoding="utf-8") as f:
        alice_text = f.read()

    # Tiền xử lý và tính n-gram
    alice_tokens = stem_lema(alice_text)
    test_tokens = stem_lema(test_text)

    alice_profile = count_word(alice_tokens, n=2)
    test_profile = count_word(test_tokens, n=2)

    # Tính Cosine Similarity
    similarity = cosine_similarity_sklearn(alice_profile, test_profile)

    return similarity


def run_streamlit():
    st.set_page_config(page_title="Author Profiling - Lewis Carroll", layout="centered")

    st.title("📄 Author Profiling - Lewis Carroll")

    st.markdown("""
    Ứng dụng kiểm tra xem văn bản bạn cung cấp có khả năng cùng tác giả với **Lewis Carroll** hay không, 
    dựa trên kỹ thuật phân tích văn bản và so sánh đặc trưng n-gram.
    """)

    st.header("🔍 Bước 1: Chọn file văn bản cần kiểm tra")
    uploaded_file = st.file_uploader("Chọn file (.txt)", type="txt")

    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')
        st.text_area("📑 Nội dung văn bản:", text, height=200)

        st.header("⚙️ Bước 2: Kết quả kiểm tra")

        if st.button("Kiểm tra"):
            similarity = compare_authors(text)

            st.write(f"**Cosine Similarity:** {similarity:.4f}")

            if similarity > THRESHOLD:
                st.success("✅ Kết luận: Có khả năng cùng tác giả Lewis Carroll.")
            else:
                st.warning("⚠️ Kết luận: Khác tác giả Lewis Carroll.")


run_streamlit()