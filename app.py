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

# UCLA Institutional Branding CSS
st.markdown("""
<style>
    /* UCLA Brand Colors */
    :root {
        --ucla-blue: #2774AE;
        --ucla-gold: #FFD100;
        --ucla-dark-blue: #005587;
        --ucla-light-blue: #8BB8E8;
        --ucla-gray: #4A4A4A;
        --ucla-light-gray: #F5F5F5;
        --ucla-white: #FFFFFF;
    }
    
    /* UCLA Typography */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* UCLA Header */
    .ucla-header {
        background: var(--ucla-blue);
        padding: 1.5rem 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 2px 8px rgba(39, 116, 174, 0.3);
    }
    
    .ucla-header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .ucla-logo {
        display: flex;
        align-items: center;
        color: white;
    }
    
    .ucla-logo-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin-left: 1rem;
    }
    
    .ucla-subtitle {
        color: var(--ucla-light-blue);
        font-size: 1rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Main Container */
    .ucla-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* UCLA Search Section */
    .ucla-search-section {
        background: var(--ucla-white);
        border: 2px solid var(--ucla-light-blue);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(39, 116, 174, 0.1);
    }
    
    .ucla-search-title {
        color: var(--ucla-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* UCLA Result Cards */
    .ucla-result-card {
        background: var(--ucla-white);
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .ucla-result-card:hover {
        border-color: var(--ucla-blue);
        box-shadow: 0 4px 12px rgba(39, 116, 174, 0.15);
    }
    
    .ucla-category-badge {
        background: var(--ucla-blue);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .ucla-question-title {
        color: var(--ucla-dark-blue);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
    
    .ucla-answer-text {
        color: var(--ucla-gray);
        line-height: 1.6;
        margin: 0;
    }
    
    .ucla-relevance-score {
        background: var(--ucla-light-gray);
        color: var(--ucla-gray);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* UCLA Sidebar */
    .ucla-sidebar-section {
        background: var(--ucla-white);
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .ucla-sidebar-title {
        color: var(--ucla-blue);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--ucla-gold);
        padding-bottom: 0.5rem;
    }
    
    .ucla-quick-btn {
        background: var(--ucla-blue);
        color: white;
        border: none;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        margin: 0.3rem 0;
        width: 100%;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .ucla-quick-btn:hover {
        background: var(--ucla-dark-blue);
    }
    
    /* UCLA Stats */
    .ucla-stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid #E0E0E0;
    }
    
    .ucla-stat-item:last-child {
        border-bottom: none;
    }
    
    .ucla-stat-label {
        font-weight: 500;
        color: var(--ucla-gray);
    }
    
    .ucla-stat-value {
        font-weight: 600;
        color: var(--ucla-blue);
        background: var(--ucla-light-gray);
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
    }
    
    /* UCLA Tips Section */
    .ucla-tips-section {
        background: var(--ucla-light-gray);
        border-left: 4px solid var(--ucla-gold);
        padding: 1.5rem;
        border-radius: 4px;
    }
    
    .ucla-tips-title {
        color: var(--ucla-blue);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .ucla-tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .ucla-tips-list li {
        padding: 0.4rem 0;
        color: var(--ucla-gray);
    }
    
    .ucla-tips-list li:before {
        content: "•";
        color: var(--ucla-blue);
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* UCLA Footer */
    .ucla-footer {
        background: var(--ucla-dark-blue);
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin: 2rem -1rem -1rem -1rem;
        border-radius: 6px 6px 0 0;
    }
    
    .ucla-footer a {
        color: var(--ucla-light-blue);
        text-decoration: none;
        font-weight: 500;
    }
    
    .ucla-footer a:hover {
        text-decoration: underline;
    }
    
    /* UCLA Category Browser */
    .ucla-category-section {
        background: var(--ucla-white);
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .ucla-category-title {
        color: var(--ucla-blue);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--ucla-gold);
        padding-bottom: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .ucla-logo-text {
            font-size: 1.5rem;
        }
        
        .ucla-container {
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
# UCLA Institutional Header
# ----------------------------
st.markdown("""
<div class="ucla-header">
    <div class="ucla-header-content">
        <div class="ucla-logo">
            <div style="font-size: 2rem; margin-right: 1rem;">🏛️</div>
            <div>
                <div class="ucla-logo-text">UCLA</div>
                <div class="ucla-subtitle">Teaching & Learning Center</div>
            </div>
        </div>
        <div style="color: white; text-align: right;">
            <div style="font-size: 1.2rem; font-weight: 600;">Knowledge Base</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">Resource Center</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Main Content
# ----------------------------
st.markdown('<div class="ucla-container">', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Search Section
    st.markdown('<div class="ucla-search-section">', unsafe_allow_html=True)
    st.markdown('<div class="ucla-search-title">🔍 Search Knowledge Base</div>', unsafe_allow_html=True)
    
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
                <div class="ucla-result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span class="ucla-category-badge">{result["category"]}</span>
                        <span class="ucla-relevance-score">Relevance: {result["relevance_score"]:.2f}</span>
                    </div>
                    <div class="ucla-question-title">{result["question"]}</div>
                    <div class="ucla-answer-text">{result["answer"]}</div>
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
            <div class="ucla-result-card">
                <span class="ucla-category-badge">{item["category"]}</span>
                <div class="ucla-question-title">{item["question"]}</div>
                <div class="ucla-answer-text">{item["answer"]}</div>
            </div>
            ''', unsafe_allow_html=True)

with col2:
    # Quick Access Section
    st.markdown('<div class="ucla-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="ucla-sidebar-title">🚀 Quick Access</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="ucla-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="ucla-sidebar-title">📊 Knowledge Base Stats</div>', unsafe_allow_html=True)
    
    total_qa = len(corpus)
    categories_count = len(categorized_corpus)
    
    st.markdown(f'''
    <div class="ucla-stat-item">
        <span class="ucla-stat-label">Total Q&A Pairs</span>
        <span class="ucla-stat-value">{total_qa}</span>
    </div>
    <div class="ucla-stat-item">
        <span class="ucla-stat-label">Categories</span>
        <span class="ucla-stat-value">{categories_count}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("**Category Breakdown:**")
    for category, items in sorted(categorized_corpus.items(), key=lambda x: len(x[1]), reverse=True):
        st.markdown(f"• {category}: {len(items)} items")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips Section
    st.markdown('<div class="ucla-tips-section">', unsafe_allow_html=True)
    st.markdown('<div class="ucla-tips-title">💡 Search Tips</div>', unsafe_allow_html=True)
    st.markdown('''
    <ul class="ucla-tips-list">
        <li>Use specific keywords for better results</li>
        <li>Try asking about SET surveys or teaching resources</li>
        <li>Browse by category to explore related content</li>
        <li>Results are ranked by relevance to your query</li>
        <li>Check quick access for urgent topics</li>
    </ul>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Category Browser Section
st.markdown('<div class="ucla-category-section">', unsafe_allow_html=True)
st.markdown('<div class="ucla-category-title">📚 Browse by Category</div>', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)

# UCLA Institutional Footer
st.markdown("""
<div class="ucla-footer">
    <strong>UCLA Teaching & Learning Center Knowledge Base</strong><br>
    <a href="https://teaching.ucla.edu" target="_blank">Visit TLC Website</a> | 
    <a href="mailto:tlc@teaching.ucla.edu">Contact TLC</a> | 
    <a href="https://ucla.edu" target="_blank">UCLA Home</a>
</div>
""", unsafe_allow_html=True)
