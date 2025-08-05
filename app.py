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

# UCLA Brand Colors Only
st.markdown("""
<style>
    /* UCLA Brand Colors Only */
    :root {
        --ucla-blue: #2774AE;
        --ucla-gold: #FFD100;
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
        box-shadow: 0 4px 12px rgba(39, 116, 174, 0.3);
    }
    
    .ucla-header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        text-align: center;
        color: white;
    }
    
    .ucla-header-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .ucla-header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Main Container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Card Layout */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .ucla-card {
        background: var(--ucla-white);
        border: 2px solid var(--ucla-blue);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(39, 116, 174, 0.15);
        transition: all 0.3s ease;
    }
    
    .ucla-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(39, 116, 174, 0.25);
    }
    
    .card-title {
        color: var(--ucla-blue);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--ucla-gold);
        padding-bottom: 0.5rem;
    }
    
    .card-content {
        color: #333;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Search Section */
    .search-section {
        background: var(--ucla-white);
        border: 2px solid var(--ucla-blue);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(39, 116, 174, 0.15);
    }
    
    .search-title {
        color: var(--ucla-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* UCLA Gold Search Button */
    .stButton > button {
        background-color: var(--ucla-gold) !important;
        color: #333 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #E6C200 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Result Cards */
    .result-card {
        background: var(--ucla-white);
        border: 1px solid var(--ucla-blue);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(39, 116, 174, 0.1);
    }
    
    .category-badge {
        background: var(--ucla-blue);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .question-title {
        color: var(--ucla-blue);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .answer-text {
        color: #333;
        line-height: 1.4;
        font-size: 0.9rem;
    }
    
    /* Consistent Font Sizes */
    .category-item {
        font-size: 0.9rem;
        margin: 0.3rem 0;
        color: #333;
    }
    
    .tips-item {
        font-size: 0.9rem;
        margin: 0.3rem 0;
        color: #333;
    }
    
    /* UCLA Footer */
    .ucla-footer {
        background: var(--ucla-blue);
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin: 2rem -1rem -1rem -1rem;
        border-radius: 8px 8px 0 0;
    }
    
    .ucla-footer a {
        color: var(--ucla-gold);
        text-decoration: none;
        font-weight: 500;
    }
    
    .ucla-footer a:hover {
        text-decoration: underline;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .card-grid {
            grid-template-columns: 1fr;
        }
        
        .ucla-header-title {
            font-size: 1.5rem;
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
# UCLA Header
# ----------------------------
st.markdown("""
<div class="ucla-header">
    <div class="ucla-header-content">
        <div class="ucla-header-title">UCLA Teaching & Learning Center</div>
        <div class="ucla-header-subtitle">Knowledge Base & Resource Center</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Main Container
# ----------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ----------------------------
# Search Section
# ----------------------------
st.markdown('<div class="search-section">', unsafe_allow_html=True)
st.markdown('<div class="search-title">Search Knowledge Base</div>', unsafe_allow_html=True)

# Search input
query = st.text_input(
    "Enter your question:",
    value=st.session_state.get("quick_search", ""),
    placeholder="e.g., How do I interpret SET survey results?"
)

search_button = st.button("Search", use_container_width=True)

# Search results
if search_button and query.strip():
    with st.spinner("Searching our knowledge base..."):
        results = enhanced_search(query, model, corpus, index, k=5)
    
    if results:
        st.markdown(f"### Search Results ({len(results)} found)")
        
        for result in results:
            st.markdown(f'''
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <span class="category-badge">{result["category"]}</span>
                    <small style="color: #666;">Relevance: {result["relevance_score"]:.2f}</small>
                </div>
                <div class="question-title">{result["question"]}</div>
                <div class="answer-text">{result["answer"]}</div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.warning("No relevant results found. Try rephrasing your question or browse by category.")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 3x2 Card Layout
# ----------------------------
st.markdown('<div class="card-grid">', unsafe_allow_html=True)

# Card 1: Quick Access
st.markdown('''
<div class="ucla-card">
    <div class="card-title">Quick Access</div>
    <div class="card-content">
        <div style="margin-bottom: 0.5rem;">
            <button onclick="window.location.href='?quick_search=emergency+procedures'" style="background: var(--ucla-blue); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.2rem; font-size: 0.8rem; cursor: pointer;">Emergency Procedures</button>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <button onclick="window.location.href='?quick_search=FERPA+guidelines'" style="background: var(--ucla-blue); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.2rem; font-size: 0.8rem; cursor: pointer;">FERPA Guidelines</button>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <button onclick="window.location.href='?quick_search=SET+survey+interpretation'" style="background: var(--ucla-blue); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.2rem; font-size: 0.8rem; cursor: pointer;">SET Surveys</button>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <button onclick="window.location.href='?quick_search=educational+innovation+grants'" style="background: var(--ucla-blue); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.2rem; font-size: 0.8rem; cursor: pointer;">Grant Opportunities</button>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Card 2: Browse by Category
st.markdown('''
<div class="ucla-card">
    <div class="card-title">Browse by Category</div>
    <div class="card-content">
        <div class="category-item">• Emergency & Safety: 4 items</div>
        <div class="category-item">• Legal & Compliance: 7 items</div>
        <div class="category-item">• Student Support: 5 items</div>
        <div class="category-item">• Teaching Resources: 9 items</div>
        <div class="category-item">• Assessment: 35 items</div>
        <div class="category-item">• Teaching Strategies: 5 items</div>
    </div>
</div>
''', unsafe_allow_html=True)

# Card 3: Knowledge Base Stats
total_qa = len(corpus)
categories_count = len(categorized_corpus)

st.markdown(f'''
<div class="ucla-card">
    <div class="card-title">Knowledge Base Stats</div>
    <div class="card-content">
        <div style="margin-bottom: 1rem;">
            <strong>Total Q&A Pairs:</strong> {total_qa}
        </div>
        <div style="margin-bottom: 1rem;">
            <strong>Categories:</strong> {categories_count}
        </div>
        <div style="font-size: 0.8rem; color: #666;">
            Most comprehensive UCLA teaching resource database
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Card 4: Search Tips
st.markdown('''
<div class="ucla-card">
    <div class="card-title">Search Tips</div>
    <div class="card-content">
        <div class="tips-item">• Use specific keywords for better results</div>
        <div class="tips-item">• Try asking about SET surveys or teaching resources</div>
        <div class="tips-item">• Browse by category to explore related content</div>
        <div class="tips-item">• Results are ranked by relevance to your query</div>
        <div class="tips-item">• Check quick access for urgent topics</div>
    </div>
</div>
''', unsafe_allow_html=True)

# Card 5: Popular Topics
st.markdown('''
<div class="ucla-card">
    <div class="card-title">Popular Topics</div>
    <div class="card-content">
        <div class="category-item">• SET Survey Interpretation</div>
        <div class="category-item">• FERPA Compliance</div>
        <div class="category-item">• Emergency Procedures</div>
        <div class="category-item">• Student Mental Health</div>
        <div class="category-item">• Teaching Resources</div>
        <div class="category-item">• Grant Opportunities</div>
    </div>
</div>
''', unsafe_allow_html=True)

# Card 6: Contact & Support
st.markdown('''
<div class="ucla-card">
    <div class="card-title">Contact & Support</div>
    <div class="card-content">
        <div style="margin-bottom: 0.5rem;">
            <strong>Email:</strong> tlc@teaching.ucla.edu
        </div>
        <div style="margin-bottom: 0.5rem;">
            <strong>Website:</strong> teaching.ucla.edu
        </div>
        <div style="margin-bottom: 0.5rem;">
            <strong>Hours:</strong> Mon-Fri 9AM-5PM
        </div>
        <div style="font-size: 0.8rem; color: #666;">
            Get personalized support for your teaching needs
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# UCLA Footer
# ----------------------------
st.markdown("""
<div class="ucla-footer">
    <strong>UCLA Teaching & Learning Center Knowledge Base</strong><br>
    <a href="https://teaching.ucla.edu" target="_blank">Visit TLC Website</a> | 
    <a href="mailto:tlc@teaching.ucla.edu">Contact TLC</a> | 
    <a href="https://ucla.edu" target="_blank">UCLA Home</a>
</div>
""", unsafe_allow_html=True)
