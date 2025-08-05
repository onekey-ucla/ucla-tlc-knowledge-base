import streamlit as st
from sentence_transformers import SentenceTransformer
from retrieve import retrieve_answer, load_index, load_corpus
import json
from collections import defaultdict
import numpy as np

# Updated: 2025-08-05 - Enhanced corpus with SET survey content

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="UCLA Teaching & Learning Center Knowledge Base", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Reset and base styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Professional header */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Main content container */
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Professional search box */
    .search-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .search-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
    }
    
    /* Enhanced result cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8f0fe;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .question-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
    
    .answer-text {
        color: #4a5568;
        line-height: 1.6;
        margin: 0;
    }
    
    .relevance-score {
        background: #f7fafc;
        color: #4a5568;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Stats cards */
    .stats-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .stats-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stat-item:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        font-weight: 500;
        color: #4a5568;
    }
    
    .stat-value {
        font-weight: 600;
        color: #1e3c72;
        background: #e8f0fe;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
    }
    
    /* Quick access buttons */
    .quick-access-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .quick-access-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
    }
    
    .quick-access-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 500;
        margin: 0.3rem 0;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .quick-access-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Tips section */
    .tips-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .tips-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tips-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    
    .tips-list li:last-child {
        border-bottom: none;
    }
    
    /* Footer */
    .footer {
        background: #1e3c72;
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin: 2rem -1rem -1rem -1rem;
        border-radius: 15px 15px 0 0;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .content-container {
            padding: 0 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    index = load_index("faiss_index.bin")
    corpus = load_corpus("corpus.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return index, corpus, model

index, corpus, model = load_data()

# ----------------------------
# Categorize corpus
# ----------------------------
def categorize_qa(corpus):
    categories = defaultdict(list)
    for item in corpus:
        # Use pre-defined category if available, otherwise fall back to keyword matching
        if "category" in item and item["category"]:
            category = item["category"]
        else:
            # Fallback categorization based on keywords
            question_lower = item["question"].lower()
            if any(word in question_lower for word in ["emergency", "safety", "crisis", "threat", "evacuation"]):
                category = "Emergency & Safety"
            elif any(word in question_lower for word in ["ferpa", "legal", "compliance", "privacy", "title ix"]):
                category = "Legal & Compliance"
            elif any(word in question_lower for word in ["grant", "funding", "innovation", "educational"]):
                category = "Grants & Funding"
            elif any(word in question_lower for word in ["student", "mental health", "counseling", "wellbeing"]):
                category = "Student Support"
            elif any(word in question_lower for word in ["accessibility", "digital", "wcag", "inclusive"]):
                category = "Accessibility & Inclusion"
            elif any(word in question_lower for word in ["feedback", "survey", "evaluation", "reflection"]):
                category = "Teaching Improvement"
            elif any(word in question_lower for word in ["syllabus", "course", "teaching", "classroom"]):
                category = "Teaching Resources"
            else:
                category = "General"
        
        item["category"] = category
        categories[category].append(item)
    
    return dict(categories)

categorized_corpus = categorize_qa(corpus)

# ----------------------------
# Enhanced search function
# ----------------------------
def enhanced_search(query, model, corpus, index, k=5):
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(corpus):
            results.append({
                "question": corpus[idx]["question"],
                "answer": corpus[idx]["answer"],
                "category": corpus[idx].get("category", "General"),
                "relevance_score": float(score),
                "rank": i + 1
            })
    
    return results

# ----------------------------
# Professional Header
# ----------------------------
st.markdown("""
<div class="header-container">
    <div class="header-content">
        <div class="header-title">UCLA Teaching & Learning Center</div>
        <div class="header-subtitle">Knowledge Base & Resource Center</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Main Content
# ----------------------------
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Search Section
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="search-title">🔍 Search Knowledge Base</div>', unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get("quick_search", ""),
        placeholder="e.g., How do I interpret SET survey results?"
    )
    
    search_button = st.button("Search", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search results
    if search_button and query.strip():
        with st.spinner("Searching our knowledge base..."):
            results = enhanced_search(query, model, corpus, index, k=5)
        
        if results:
            st.markdown(f"### 📋 Search Results ({len(results)} found)")
            
            for result in results:
                st.markdown(f'''
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span class="category-badge">{result["category"]}</span>
                        <span class="relevance-score">Relevance: {result["relevance_score"]:.2f}</span>
                    </div>
                    <div class="question-title">{result["question"]}</div>
                    <div class="answer-text">{result["answer"]}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("No relevant results found. Try rephrasing your question or browse by category.")
    
    # Show category content if selected
    elif st.session_state.get("selected_category", "All Categories") != "All Categories":
        selected_category = st.session_state.get("selected_category")
        st.markdown(f"### 📚 {selected_category}")
        
        for item in categorized_corpus.get(selected_category, []):
            st.markdown(f'''
            <div class="result-card">
                <span class="category-badge">{item["category"]}</span>
                <div class="question-title">{item["question"]}</div>
                <div class="answer-text">{item["answer"]}</div>
            </div>
            ''', unsafe_allow_html=True)

with col2:
    # Quick Access Section
    st.markdown('<div class="quick-access-container">', unsafe_allow_html=True)
    st.markdown('<div class="quick-access-title">🚀 Quick Access</div>', unsafe_allow_html=True)
    
    if st.button("🚨 Emergency Procedures", use_container_width=True, key="emergency"):
        st.session_state.quick_search = "emergency procedures"
        st.rerun()
    
    if st.button("📋 FERPA Guidelines", use_container_width=True, key="ferpa"):
        st.session_state.quick_search = "FERPA guidelines"
        st.rerun()
    
    if st.button("📊 SET Surveys", use_container_width=True, key="set"):
        st.session_state.quick_search = "SET survey interpretation"
        st.rerun()
    
    if st.button("💰 Grant Opportunities", use_container_width=True, key="grants"):
        st.session_state.quick_search = "educational innovation grants"
        st.rerun()
    
    if st.button("🎓 Student Support", use_container_width=True, key="student"):
        st.session_state.quick_search = "student mental health support"
        st.rerun()
    
    if st.button("🏫 Teaching Resources", use_container_width=True, key="resources"):
        st.session_state.quick_search = "UCLA Teaching and Learning Center"
        st.rerun()
    
    if st.button("📈 Response Rates", use_container_width=True, key="rates"):
        st.session_state.quick_search = "SET survey response rates"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    st.markdown('<div class="stats-title">📊 Knowledge Base Stats</div>', unsafe_allow_html=True)
    
    total_qa = len(corpus)
    categories_count = len(categorized_corpus)
    
    st.markdown(f'''
    <div class="stat-item">
        <span class="stat-label">Total Q&A Pairs</span>
        <span class="stat-value">{total_qa}</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">Categories</span>
        <span class="stat-value">{categories_count}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("**Category Breakdown:**")
    for category, items in sorted(categorized_corpus.items(), key=lambda x: len(x[1]), reverse=True):
        st.markdown(f"• {category}: {len(items)} items")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips Section
    st.markdown('<div class="tips-container">', unsafe_allow_html=True)
    st.markdown('<div class="tips-title">💡 Search Tips</div>', unsafe_allow_html=True)
    st.markdown('''
    <ul class="tips-list">
        <li>Use specific keywords for better results</li>
        <li>Try asking about SET surveys or teaching resources</li>
        <li>Browse by category to explore related content</li>
        <li>Results are ranked by relevance to your query</li>
        <li>Check quick access for urgent topics</li>
    </ul>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Category Browser Section
st.markdown("---")
st.markdown("### 📚 Browse by Category")

# Create category selection
category_options = ["All Categories"] + sorted(categorized_corpus.keys())
selected_category = st.selectbox(
    "Choose a category to browse:",
    category_options,
    key="category_selector"
)

if selected_category != "All Categories":
    st.session_state.selected_category = selected_category
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Professional Footer
st.markdown("""
<div class="footer">
    <strong>UCLA Teaching & Learning Center Knowledge Base</strong><br>
    <a href="https://teaching.ucla.edu" target="_blank">Visit TLC Website</a> | 
    <a href="mailto:tlc@teaching.ucla.edu">Contact TLC</a>
</div>
""", unsafe_allow_html=True)
