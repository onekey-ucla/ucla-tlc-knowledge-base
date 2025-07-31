# UCLA TLC Knowledge Base

A web-based knowledge base for UCLA instructors to find answers about teaching policies, procedures, and resources.

## Features

- 🔍 **Semantic Search**: Find relevant answers using natural language
- 📚 **Category Browsing**: Browse content by topic categories
- 🚨 **Quick Access**: One-click access to urgent topics
- 📊 **Enhanced UI**: Modern, responsive web interface
- 🎯 **Multiple Results**: Get multiple relevant answers instead of just one

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/ucla-tlc-knowledge-base.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the path to your app: `app.py`
   - Click "Deploy"

### Option 2: Heroku

1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Railway

1. **Create railway.json**:
   ```json
   {
     "build": {
       "builder": "NIXPACKS"
     }
   }
   ```

2. **Deploy**:
   - Connect your GitHub repo to Railway
   - Railway will auto-deploy

### Option 4: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## File Structure

```
├── app.py                 # Main Streamlit application
├── retrieve.py            # Search and retrieval functions
├── embed_index.py         # Index creation script
├── enhanced_corpus.jsonl  # Q&A data with metadata
├── requirements.txt       # Python dependencies
├── .streamlit/config.toml # Streamlit configuration
└── README.md             # This file
```

## Data Management

To update the knowledge base:

1. **Add new Q&A pairs** to `enhanced_corpus.jsonl`
2. **Regenerate the index**:
   ```bash
   python embed_index.py
   ```
3. **Redeploy** the application

## Categories

- 🚨 **Emergency & Safety**: Crisis procedures, evacuation protocols
- ⚖️ **Legal & Compliance**: FERPA, Title IX, privacy requirements
- 🎓 **Student Support**: Mental health, accommodations, wellbeing
- ♿ **Accessibility & Inclusion**: Digital accessibility, inclusive teaching
- 💰 **Grants & Funding**: Educational innovation grants
- 📚 **Teaching Resources**: Course tools, syllabi, classroom management
- 🔄 **Teaching Improvement**: Feedback, reflection, professional development
- 👥 **Professional Development**: Workshops, training, consultations

## Customization

- **Add new categories**: Update the categorization logic in `app.py`
- **Modify styling**: Edit the CSS in the `st.markdown()` section
- **Add features**: Extend the sidebar or main content areas
- **Integrate APIs**: Add external data sources or services 