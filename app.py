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
    page_title="🐻 UCLA TLC Knowledge Base", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2774AE 0%, #005587 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .search-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2774AE;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .category-tag {
        background: #2774AE;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .priority-high {
        border-left-color: #dc3545;
    }
    .priority-medium {
        border-left-color: #ffc107;
    }
    .priority-low {
        border-left-color: #28a745;
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
# Main Interface
# ----------------------------
st.markdown('<div class="main-header"><h1>🐻 UCLA TLC Knowledge Base</h1><p>Find answers to teaching questions, policies, and resources</p></div>', unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.header("📚 Browse by Category")
    
    # Category selection
    selected_category = st.selectbox(
        "Choose a category:",
        ["All Categories"] + list(categorized_corpus.keys())
    )
    
    st.markdown("---")
    st.header("🚨 Quick Access")
    
    # Quick access buttons for urgent topics
    if st.button("🚨 Emergency Procedures", use_container_width=True):
        st.session_state.quick_search = "emergency procedures"
    
    if st.button("📋 FERPA Guidelines", use_container_width=True):
        st.session_state.quick_search = "FERPA guidelines"
    
    if st.button("📊 SET Surveys", use_container_width=True):
        st.session_state.quick_search = "SET survey interpretation"
    
    if st.button("💰 Grant Opportunities", use_container_width=True):
        st.session_state.quick_search = "educational innovation grants"
    
    if st.button("🎓 Student Support", use_container_width=True):
        st.session_state.quick_search = "student mental health support"
    
    if st.button("🏫 Teaching Resources", use_container_width=True):
        st.session_state.quick_search = "UCLA Teaching and Learning Center"
    
    if st.button("📈 Response Rates", use_container_width=True):
        st.session_state.quick_search = "SET survey response rates"

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.subheader("🔍 Search Knowledge Base")
    
    # Search input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get("quick_search", ""),
        placeholder="e.g., How do I interpret SET survey results?"
    )
    
    search_button = st.button("Search", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search results
    if search_button and query.strip():
        with st.spinner("Searching..."):
            results = enhanced_search(query, model, corpus, index, k=5)
        
        if results:
            st.subheader(f"📋 Search Results ({len(results)} found)")
            
            for result in results:
                # Determine priority based on category
                priority_class = "priority-high" if result["category"] in ["Emergency & Safety", "Legal & Compliance"] else "priority-medium"
                
                st.markdown(f'''
                <div class="result-card {priority_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="category-tag">{result["category"]}</span>
                        <small style="color: #666;">Relevance: {result["relevance_score"]:.2f}</small>
                    </div>
                    <h4 style="margin: 0.5rem 0;">{result["question"]}</h4>
                    <p style="margin: 0;">{result["answer"]}</p>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("No relevant results found. Try rephrasing your question.")
    
    # Show category content if selected
    elif selected_category != "All Categories":
        st.subheader(f"📚 {selected_category}")
        
        for item in categorized_corpus[selected_category]:
            st.markdown(f'''
            <div class="result-card">
                <span class="category-tag">{item["category"]}</span>
                <h4 style="margin: 0.5rem 0;">{item["question"]}</h4>
                <p style="margin: 0;">{item["answer"]}</p>
            </div>
            ''', unsafe_allow_html=True)

with col2:
    st.subheader("📊 Quick Stats")
    
    # Statistics
    total_qa = len(corpus)
    categories_count = len(categorized_corpus)
    
    st.metric("Total Q&A Pairs", total_qa)
    st.metric("Categories", categories_count)
    
    # Category breakdown with updated counts
    st.markdown("**Category Breakdown:**")
    for category, items in categorized_corpus.items():
        st.markdown(f"- {category}: {len(items)} items")
    
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    - Use specific keywords for better results
    - Check the sidebar for quick access to urgent topics
    - Browse by category to explore related content
    - Results are ranked by relevance to your query
    - Try asking about SET surveys, teaching resources, or compliance
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    UCLA Teaching and Learning Center Knowledge Base | 
    <a href="https://teaching.ucla.edu" target="_blank">Visit TLC Website</a>
</div>
""", unsafe_allow_html=True)
