import streamlit as st
from sentence_transformers import SentenceTransformer
from retrieve import retrieve_answer, load_index, load_corpus
import json
from collections import defaultdict
import numpy as np

# Updated: 2025-08-05 - Enhanced corpus with SET survey content - CLEAN OUTLINE

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
    
    /* Section Styling */
    .section-header {
        color: var(--ucla-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--ucla-gold);
    }
    
    .subsection-header {
        color: var(--ucla-blue);
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
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
# SECTION 1: Search Knowledge Base
# ----------------------------
st.markdown('<div class="section-header">🔍 Search Knowledge Base</div>', unsafe_allow_html=True)

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

# ----------------------------
# SECTION 2: Quick Access
# ----------------------------
st.markdown('<div class="section-header">⚡ Quick Access</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("Emergency Procedures", key="emergency", use_container_width=True)

with col2:
    st.button("FERPA Guidelines", key="ferpa", use_container_width=True)

with col3:
    st.button("SET Surveys", key="set", use_container_width=True)

with col4:
    st.button("Grant Opportunities", key="grants", use_container_width=True)

# ----------------------------
# SECTION 3: Knowledge Base Overview
# ----------------------------
st.markdown('<div class="section-header">📊 Knowledge Base Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_qa = len(corpus)
    categories_count = len(categorized_corpus)
    st.markdown('<div class="subsection-header">Statistics</div>', unsafe_allow_html=True)
    st.metric("Total Q&A Pairs", total_qa)
    st.metric("Categories", categories_count)

with col2:
    st.markdown('<div class="subsection-header">Category Breakdown</div>', unsafe_allow_html=True)
    st.markdown("• Emergency & Safety: 4 items")
    st.markdown("• Legal & Compliance: 7 items")
    st.markdown("• Student Support: 5 items")
    st.markdown("• Teaching Resources: 9 items")
    st.markdown("• Assessment: 35 items")
    st.markdown("• Teaching Strategies: 5 items")

# ----------------------------
# SECTION 4: Browse by Category
# ----------------------------
st.markdown('<div class="section-header">📂 Browse by Category</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="subsection-header">Teaching & Learning</div>', unsafe_allow_html=True)
    st.markdown("• Teaching Resources")
    st.markdown("• Teaching Strategies")
    st.markdown("• Assessment & Evaluation")

with col2:
    st.markdown('<div class="subsection-header">Student Support</div>', unsafe_allow_html=True)
    st.markdown("• Student Mental Health")
    st.markdown("• Accessibility & Inclusion")
    st.markdown("• Emergency & Safety")

with col3:
    st.markdown('<div class="subsection-header">Administrative</div>', unsafe_allow_html=True)
    st.markdown("• Legal & Compliance")
    st.markdown("• Grants & Funding")
    st.markdown("• SET Surveys")

# ----------------------------
# SECTION 5: Popular Topics
# ----------------------------
st.markdown('<div class="section-header">🔥 Popular Topics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**SET Survey Interpretation**")
    st.markdown("*Understanding and using SET results effectively*")

with col2:
    st.markdown("**FERPA Compliance**")
    st.markdown("*Student privacy and data protection guidelines*")

with col3:
    st.markdown("**Emergency Procedures**")
    st.markdown("*Crisis response and safety protocols*")

with col4:
    st.markdown("**Teaching Resources**")
    st.markdown("*Syllabus design and classroom strategies*")

# ----------------------------
# SECTION 6: Search Tips
# ----------------------------
st.markdown('<div class="section-header">💡 Search Tips</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**For Better Results:**")
    st.markdown("• Use specific keywords")
    st.markdown("• Try asking about SET surveys")
    st.markdown("• Include context in your question")

with col2:
    st.markdown("**Browse Effectively:**")
    st.markdown("• Check quick access for urgent topics")
    st.markdown("• Browse by category to explore")
    st.markdown("• Results are ranked by relevance")

# ----------------------------
# SECTION 7: Contact & Support
# ----------------------------
st.markdown('<div class="section-header">📞 Contact & Support</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Email:**")
    st.markdown("tlc@teaching.ucla.edu")

with col2:
    st.markdown("**Website:**")
    st.markdown("teaching.ucla.edu")

with col3:
    st.markdown("**Hours:**")
    st.markdown("Mon-Fri 9AM-5PM")

# ----------------------------
# UCLA Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div class="ucla-footer">
    <strong>UCLA Teaching & Learning Center Knowledge Base</strong><br>
    <a href="https://teaching.ucla.edu" target="_blank">Visit TLC Website</a> | 
    <a href="mailto:tlc@teaching.ucla.edu">Contact TLC</a> | 
    <a href="https://ucla.edu" target="_blank">UCLA Home</a>
</div>
""", unsafe_allow_html=True)
