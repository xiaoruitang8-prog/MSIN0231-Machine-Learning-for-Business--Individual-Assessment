"""
Market Research Assistant - Streamlit Application
MSIN0231 Machine Learning for Business - Individual Assignment

Uses LangChain's WikipediaRetriever as suggested in the assignment hints.
"""

import re
import json
import urllib.parse
import urllib.request
from typing import List, Dict, Tuple

import streamlit as st

# --- Check for OpenAI ---
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

# --- Check for LangChain Wikipedia ---
LANGCHAIN_WIKI_AVAILABLE = False
try:
    from langchain_community.retrievers import WikipediaRetriever
    LANGCHAIN_WIKI_AVAILABLE = True
except ImportError:
    pass


# ==================== LLM API Functions ====================
def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash-exp", max_tokens: int = 1000) -> str:
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        
        candidates = result.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else ""
        raise Exception(f"Gemini API Error: {error_body[:200]}")
    except Exception as e:
        raise Exception(f"Gemini Error: {str(e)}")


# ==================== Wikipedia Functions (using LangChain WikipediaRetriever) ====================
def get_wikipedia_pages(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Get k relevant Wikipedia pages using LangChain's WikipediaRetriever.
    As suggested in assignment hints: https://python.langchain.com/docs/integrations/retrievers/wikipedia/
    
    Uses multiple search strategies to ensure we get exactly k relevant pages.
    """
    if not LANGCHAIN_WIKI_AVAILABLE:
        raise ImportError("LangChain Wikipedia not installed. Run: pip install langchain-community wikipedia")
    
    # Initialize WikipediaRetriever with more results to filter
    retriever = WikipediaRetriever(
        top_k_results=k + 5,  # Get extra results to filter irrelevant ones
        doc_content_chars_max=800
    )
    
    # Multiple search queries to ensure we get enough relevant results
    # Added more query variants to improve coverage
    search_queries = [
        f"{query} industry",
        query,
        f"{query} market",
        f"{query} production",
        f"{query} business",
        f"{query} sector",
        f"{query} economy",
        f"{query} manufacturing",
    ]
    
    pages = []
    seen_titles = set()
    
    # Keywords that indicate irrelevant results (people, disambiguation, etc.)
    # Reduced patterns to avoid filtering out legitimate pages
    irrelevant_patterns = [
        "(surname)", "(name)", "(disambiguation)", "(film)", "(song)", 
        "(band)", "(album)", "(tv series)", "(novel)"
    ]
    
    for search_query in search_queries:
        if len(pages) >= k:
            break
            
        try:
            # Version-safe retriever call with fallback
            try:
                docs = retriever.invoke(search_query)
            except AttributeError:
                # Fallback for older LangChain versions
                docs = retriever.get_relevant_documents(search_query)
            
            for doc in docs:
                if len(pages) >= k:
                    break
                    
                title = doc.metadata.get("title", "Unknown")
                title_lower = title.lower()
                
                # Skip duplicates
                if title_lower in seen_titles:
                    continue
                
                # Skip irrelevant pages (people, films, songs, etc.)
                is_irrelevant = any(pattern in title_lower for pattern in irrelevant_patterns)
                if is_irrelevant:
                    continue
                
                seen_titles.add(title_lower)
                
                # Get URL from metadata or construct it
                url = doc.metadata.get("source", "")
                if not url:
                    encoded_title = urllib.parse.quote(title.replace(' ', '_'))
                    url = f"https://en.wikipedia.org/wiki/{encoded_title}"
                
                # Get excerpt from page content
                excerpt = doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content
                
                pages.append({
                    "title": title,
                    "url": url,
                    "excerpt": excerpt
                })
        except Exception:
            continue
    
    return pages


# ==================== Validation Functions ====================
def is_valid_industry_basic(text: str) -> Tuple[bool, str]:
    """Basic validation - checks for obvious invalid input without requiring API."""
    text = text.strip()
    
    # Too short
    if len(text) < 2:
        return False, "Please enter at least 2 characters."
    
    # Must have at least one letter
    if not any(c.isalpha() for c in text):
        return False, "Please include letters in your input."
    
    # Obvious keyboard mashing patterns
    keyboard_patterns = ['qwerty', 'asdfgh', 'zxcvbn', 'qwertz', 'hjkl', 'abc', 'xyz']
    text_lower = text.lower().replace(' ', '')
    for pattern in keyboard_patterns:
        if text_lower == pattern:  # Exact match
            return False, "Please enter a valid industry."
    
    # Repeated characters (like "aaaa" or "aaa")
    if re.search(r'(.)\1{2,}', text):
        return False, "Please enter a valid industry."
    
    # Must contain at least one vowel (real English words have vowels)
    # Exception: Short abbreviations like "AI", "TV", "IT" are allowed
    vowels = set('aeiouAEIOU')
    has_vowel = any(c in vowels for c in text)
    if len(text) > 2 and not has_vowel:
        return False, "Please enter a valid industry name."
    
    # Reject very short random strings without vowels
    if len(text) <= 4:
        text_alpha = ''.join(c for c in text if c.isalpha()).lower()
        # Known valid short terms
        valid_short = {'ai', 'it', 'tv', 'oil', 'gas', 'tea', 'car', 'food', 'tech', 
                       'wine', 'beer', 'coal', 'gold', 'iron', 'wood', 'film', 'art', 
                       'law', 'toy', 'pet', 'spa', 'rum', 'gin', 'ice', 'egg', 'ham'}
        if len(text_alpha) <= 4 and text_alpha not in valid_short and not has_vowel:
            return False, "Please enter a valid industry name."
    
    return True, ""


# ==================== Report Functions ====================
def generate_report(industry: str, pages: List[Dict], api_key: str, model: str = "gpt-4o-mini", provider: str = "openai") -> str:
    """Generate industry report using LLM."""
    
    # Build sources text from Wikipedia pages
    sources = "\n"
    for i, p in enumerate(pages, 1):
        excerpt = p.get('excerpt', '') or '(No excerpt available)'
        sources += f"[Source {i}] {p['title']}:\n{excerpt}\n\n"
    
    system_prompt = """You are a market research analyst. Write a professional industry/market report.

REQUIREMENTS:
- MUST be between 400-490 words (this is critical! Do NOT exceed 490 words!)
- Professional, analytical tone
- Structure with these sections:
  * Overview (industry definition, scope, significance)
  * Key Players & Segments (major companies, market segments)
  * Market Drivers (demand factors, customer needs)
  * Competitive Landscape (competition, market structure)
  * Trends & Outlook (current trends, future prospects)
- Base analysis on the provided Wikipedia sources
- Be factual and informative"""

    user_prompt = f"Write an industry/market report about: {industry}\n\nWikipedia Sources:{sources}"
    
    try:
        if provider == "google":
            # Use Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            return call_gemini(full_prompt, api_key, model, max_tokens=1200)
        else:
            # Use OpenAI
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed. Run: pip install openai")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Report generation failed: {str(e)}")


def shorten_report(report: str, industry: str, api_key: str, model: str = "gpt-4o-mini", provider: str = "openai") -> str:
    """Shorten report to under 500 words while preserving key content."""
    
    system_prompt = """You are an editor. Shorten the following report to be between 420-480 words.
Keep the same structure and key points, but make it more concise.
Do NOT add any preamble or explanation - just output the shortened report."""

    user_prompt = f"Shorten this {industry} industry report to under 490 words:\n\n{report}"
    
    try:
        if provider == "google":
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            return call_gemini(full_prompt, api_key, model, max_tokens=1200)
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed.")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Report shortening failed: {str(e)}")


def word_count(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))


# ==================== Streamlit UI ====================
st.set_page_config(page_title="Market Research Assistant", page_icon="📊", layout="wide")

# Session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'industry' not in st.session_state:
    st.session_state.industry = ""
if 'pages' not in st.session_state:
    st.session_state.pages = []
if 'report' not in st.session_state:
    st.session_state.report = ""

st.title("📊 Market Research Assistant")
st.markdown("*Generate industry reports using Wikipedia data and LLM analysis*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Select LLM Model",
        [
            "gpt-4o-mini (OpenAI - Fast & Cheap)",
            "gpt-4o (OpenAI - Better Quality)",
            "gemini-2.0-flash (Google - Free)",
            "gemini-1.5-flash (Google - Free)"
        ],
        index=0,
        help="Select your preferred LLM model. You'll need to provide your own API key."
    )
    
    # Determine provider and model
    if "gemini" in model_choice.lower():
        selected_provider = "google"
        if "2.0-flash" in model_choice:
            selected_model = "gemini-2.0-flash-exp"
        else:
            selected_model = "gemini-1.5-flash"
    else:
        selected_provider = "openai"
        if "gpt-4o-mini" in model_choice:
            selected_model = "gpt-4o-mini"
        else:
            selected_model = "gpt-4o"
    
    st.caption(f"📌 Using: {selected_model}")
    
    st.divider()
    st.markdown("### 🔑 API Key")
    st.caption("💡 API key required for report generation (Q3)")
    
    # API key input based on provider
    if selected_provider == "google":
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            help="Required for Gemini models (needed for Q3)"
        )
        if api_key_input.strip():
            api_key = api_key_input.strip()
            st.success("✅ Google API key ready")
        else:
            api_key = ""
            st.info("ℹ️ Enter API key before generating report")
            st.markdown("[Get free key →](https://aistudio.google.com/app/apikey)")
    else:
        # OpenAI - user must provide their own key
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for GPT models (needed for Q3)"
        )
        if api_key_input.strip():
            api_key = api_key_input.strip()
            st.success("✅ OpenAI API key ready")
        else:
            api_key = ""
            st.info("ℹ️ Enter API key before generating report")
            st.markdown("[Get API key →](https://platform.openai.com/api-keys)")
    
    st.divider()
    st.header("📍 Progress")
    
    # Calculate progress: Q1=33%, Q2=66%, Q3=100%
    if st.session_state.report:
        progress = 1.0
    elif st.session_state.step > 2:
        progress = 0.66
    elif st.session_state.step > 1:
        progress = 0.33
    else:
        progress = 0.0
    
    st.progress(progress)
    
    if st.session_state.report:
        st.caption("Complete! ✓")
    else:
        st.caption(f"Step {st.session_state.step} of 3")
    
    if st.session_state.step > 1:
        st.success("✓ Q1: Validated")
    if st.session_state.step > 2:
        st.success("✓ Q2: URLs retrieved")
    if st.session_state.report:
        st.success("✓ Q3: Report generated")
    
    st.divider()
    
    # Show library status
    with st.expander("📚 Library Status"):
        if OPENAI_AVAILABLE:
            st.success("✓ OpenAI")
        else:
            st.error("✗ OpenAI (pip install openai)")
        
        if LANGCHAIN_WIKI_AVAILABLE:
            st.success("✓ LangChain WikipediaRetriever")
        else:
            st.error("✗ LangChain Wikipedia (pip install langchain-community wikipedia)")
    
    st.divider()
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.step = 1
        st.session_state.industry = ""
        st.session_state.pages = []
        st.session_state.report = ""
        st.rerun()

st.divider()

# ==================== Q1: Industry Validation ====================
st.header("Q1: Industry Input Validation")
st.markdown("*Enter any industry to research.*")

industry_input = st.text_input(
    "Industry",
    placeholder="e.g., Electric Vehicles, Fruit, AI, Renewable Energy, Chocolate, Fashion"
)

with st.expander("💡 Examples of valid inputs"):
    st.markdown("""
    - **Industries:** Automotive, Healthcare, Technology, Agriculture
    - **Products:** Fruit, Cars, Smartphones, Chocolate, Wine
    - **Markets:** Luxury Goods, Fast Food, E-commerce, Streaming
    - **Sectors:** Renewable Energy, Fintech, Biotechnology
    - **Topics:** AI, Blockchain, Sustainability, Cloud Computing
    """)

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("✓ Validate", type="primary", use_container_width=True):
        if not industry_input.strip():
            st.error("❌ Please enter an industry.")
        else:
            # Basic validation only - no API key required for Q1
            valid, msg = is_valid_industry_basic(industry_input)
            if not valid:
                st.error(f"❌ {msg}")
            else:
                # Validation passed - proceed to Q2
                st.session_state.industry = industry_input.strip()
                st.session_state.pages = []
                st.session_state.report = ""
                st.session_state.step = 2
                st.success(f"✅ Industry validated: {industry_input.strip()}")
                st.rerun()

with col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.step = 1
        st.session_state.industry = ""
        st.session_state.pages = []
        st.session_state.report = ""
        st.rerun()

if st.session_state.step >= 2:
    st.success(f"✅ **Current industry:** {st.session_state.industry}")

st.divider()

# ==================== Q2: Wikipedia URLs (using LangChain WikipediaRetriever) ====================
if st.session_state.step >= 2:
    st.header("Q2: Top 5 Most Relevant Wikipedia Pages")
    
    if not st.session_state.pages:
        if st.button("🔍 Retrieve Wikipedia URLs", type="primary", use_container_width=True):
            if not LANGCHAIN_WIKI_AVAILABLE:
                st.error("❌ LangChain Wikipedia not installed. Run: `pip install langchain-community wikipedia`")
            else:
                with st.spinner("Searching Wikipedia using WikipediaRetriever..."):
                    try:
                        pages = get_wikipedia_pages(st.session_state.industry, k=5)
                        
                        if len(pages) >= 5:
                            st.session_state.pages = pages
                            st.session_state.step = 3
                            st.rerun()
                        elif len(pages) > 0:
                            st.session_state.pages = pages
                            st.session_state.step = 3
                            st.warning(f"⚠️ Found only {len(pages)} pages. Proceeding anyway.")
                            st.rerun()
                        else:
                            st.error("❌ Could not find Wikipedia pages. Please try a different industry.")
                    except Exception as e:
                        st.error(f"❌ Error retrieving Wikipedia pages: {e}")
    
    if st.session_state.pages:
        st.success(f"✅ **Top {len(st.session_state.pages)} Wikipedia Pages:**")
        for i, page in enumerate(st.session_state.pages, 1):
            with st.expander(f"**{i}. {page['title']}**", expanded=(i <= 2)):
                st.markdown(f"🔗 **URL:** [{page['url']}]({page['url']})")
                if page.get('excerpt'):
                    st.markdown(f"📄 *{page['excerpt']}*")
                else:
                    st.markdown("📄 *(No excerpt available)*")
        st.divider()

# ==================== Q3: Report Generation ====================
if st.session_state.step >= 3 and st.session_state.pages:
    st.header("Q3: Industry Report (<500 words)")
    
    if not st.session_state.report:
        if st.button("📝 Generate Report", type="primary", use_container_width=True):
            # Check API key - required for Q3
            if not api_key:
                st.error(f"❌ Please enter your {'Google' if selected_provider == 'google' else 'OpenAI'} API key in the sidebar to generate the report.")
            else:
                with st.spinner("Generating report... (this may take 10-20 seconds)"):
                    try:
                        report = generate_report(st.session_state.industry, st.session_state.pages, api_key, selected_model, selected_provider)
                        wc = word_count(report)
                        
                        # Enforce word count limit - auto-shorten if over 500
                        if wc >= 500:
                            st.info(f"📝 Report is {wc} words. Auto-shortening to under 500...")
                            report = shorten_report(report, st.session_state.industry, api_key, selected_model, selected_provider)
                            wc = word_count(report)
                            
                            # If still over, try once more
                            if wc >= 500:
                                st.info(f"📝 Still {wc} words. Shortening again...")
                                report = shorten_report(report, st.session_state.industry, api_key, selected_model, selected_provider)
                        
                        st.session_state.report = report
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
    
    if st.session_state.report:
        wc = word_count(st.session_state.report)
        st.success("✅ **Report Generated!**")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric("Words", wc)
        with col2:
            if wc < 500:
                st.success(f"✓ Within limit ({500-wc} words remaining)")
            else:
                st.warning(f"⚠️ {wc-500} words over limit")
        
        st.divider()
        st.markdown(f"## 📄 {st.session_state.industry} Industry Report")
        st.markdown("---")
        st.markdown(st.session_state.report)
        st.markdown("---")
        
        st.markdown("#### 📚 Sources (from WikipediaRetriever)")
        for i, p in enumerate(st.session_state.pages, 1):
            st.markdown(f"{i}. [{p['title']}]({p['url']})")
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📥 Download TXT",
                st.session_state.report,
                f"{st.session_state.industry.replace(' ', '_')}_report.txt",
                use_container_width=True
            )
        with col2:
            md_content = f"# {st.session_state.industry} Industry Report\n\n{st.session_state.report}\n\n## Sources\n" + "\n".join([f"- [{p['title']}]({p['url']})" for p in st.session_state.pages])
            st.download_button(
                "📥 Download MD",
                md_content,
                f"{st.session_state.industry.replace(' ', '_')}_report.md",
                use_container_width=True
            )
        with col3:
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{st.session_state.industry} Industry Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 25px; }}
        .report {{ background: #f9f9f9; padding: 25px; border-radius: 8px; margin: 20px 0; white-space: pre-wrap; }}
        .sources {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin-top: 30px; }}
        .sources h3 {{ margin-top: 0; color: #2c3e50; }}
        .sources ul {{ margin: 10px 0; padding-left: 20px; }}
        .sources a {{ color: #3498db; text-decoration: none; }}
        .sources a:hover {{ text-decoration: underline; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
        .word-count {{ text-align: right; color: #95a5a6; font-size: 0.9em; margin-top: 15px; }}
    </style>
</head>
<body>
    <h1>📊 {st.session_state.industry} Industry Report</h1>
    <p class="meta">Generated by Market Research Assistant (using LangChain WikipediaRetriever)</p>
    
    <div class="report">{st.session_state.report}</div>
    
    <p class="word-count">Word Count: {wc} words</p>
    
    <div class="sources">
        <h3>📚 Sources</h3>
        <ul>
            {"".join([f'<li><a href="{p["url"]}" target="_blank">{p["title"]}</a></li>' for p in st.session_state.pages])}
        </ul>
    </div>
</body>
</html>"""
            st.download_button(
                "📥 Download HTML",
                html_content,
                f"{st.session_state.industry.replace(' ', '_')}_report.html",
                mime="text/html",
                use_container_width=True
            )

st.divider()
st.caption("MSIN0231 Machine Learning for Business - Individual Assignment")