import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import os
import json
from agent_anp7 import HybridPortAgent

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Agent Portuaire Hybride",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Look ---
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
}
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #2a5298;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.query-box {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    margin: 1rem 0;
}
.result-box {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.sidebar-content {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1>⚓ Agent Portuaire Hybride Intelligent</h1>
    <p>Analyse avancée des données portuaires avec traitement des dates complexes</p>
</div>
""", unsafe_allow_html=True)

# --- Agent Initialization ---
@st.cache_resource(show_spinner=False)
def get_agent():
    return HybridPortAgent("anp_dataset_clean_sample.csv")

agent = get_agent()

# --- Session State for User, Q/A History and Result Cache (with file persistence) ---
def get_user_history_path(username):
    return f".chat_history_{username}.json"
def get_user_cache_path(username):
    return f".chat_cache_{username}.json"

def load_history(username):
    path = get_user_history_path(username)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(username, history):
    path = get_user_history_path(username)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass

def load_cache(username):
    path = get_user_cache_path(username)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                # Convert keys to str (json keys are always str)
                return {str(k): v for k, v in cache.items()}
        except Exception:
            return {}
    return {}

def save_cache(username, cache):
    path = get_user_cache_path(username)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass

if 'username' not in st.session_state:
    st.session_state.username = ''
if st.session_state.username:
    if 'qa_history' not in st.session_state or st.session_state.get('last_loaded_user', None) != st.session_state.username:
        st.session_state.qa_history = load_history(st.session_state.username)
        st.session_state.qa_cache = load_cache(st.session_state.username)
        st.session_state.last_loaded_user = st.session_state.username
    qa_history = st.session_state.qa_history
    qa_cache = st.session_state.qa_cache
else:
    qa_history = []
    qa_cache = {}

# --- Sidebar: Login, System Info, Templates, Data Preview ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("👤 Connexion Utilisateur")
    username = st.text_input("Nom d'utilisateur", value=st.session_state.username, key="username_input")
    if st.button("Se connecter", key="login_btn") or (not st.session_state.username and username):
        st.session_state.username = username.strip()
        # Load or create persistent history/cache for this user
        st.session_state.qa_history = load_history(st.session_state.username)
        st.session_state.qa_cache = load_cache(st.session_state.username)
        st.session_state.last_loaded_user = st.session_state.username
        st.rerun()
    if st.session_state.username:
        st.success(f"Connecté en tant que : {st.session_state.username}")
    else:
        st.warning("Veuillez entrer un nom d'utilisateur pour commencer le chat.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("📊 Informations Système")
    status = agent.get_system_status()
    st.metric("Total Escales", f"{status['data_info']['total_rows']:,}")
    st.metric("Colonnes Disponibles", status['data_info']['total_columns'])
    st.metric("Colonnes Calculées", status['data_info']['calculated_columns'])
    if status['data_info']['date_range']['start']:
        st.write("📅 **Période des données:**")
        st.write(f"Du {status['data_info']['date_range']['start'].strftime('%d/%m/%Y') if status['data_info']['date_range']['start'] else 'N/A'}")
        st.write(f"Au {status['data_info']['date_range']['end'].strftime('%d/%m/%Y') if status['data_info']['date_range']['end'] else 'N/A'}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("⚡ Performance")
    perf = status['performance']
    success_rate = (perf['successful_queries'] / perf['total_queries'] * 100) if perf['total_queries'] > 0 else 0
    st.metric("Requêtes Traitées", perf['total_queries'])
    st.metric("Taux de Succès", f"{success_rate:.1f}%")
    if perf['avg_response_time'] > 0:
        st.metric("Temps Moyen", f"{perf['avg_response_time']:.2f}s")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Area ---

# --- Chatbot UI: Q/A bubbles (oldest at top, newest at bottom), input at bottom ---
st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.header("🤖 Agent Portuaire - Chatbot")

# Display all Q/A pairs as chat bubbles (oldest at top)
if st.session_state.username:
    # Show current question being analyzed (if any)
    if 'current_question' in st.session_state and st.session_state.current_question:
        st.markdown(f"<div style='background:#fffbe6;padding:0.7em 1em;border-radius:10px;margin-bottom:0.5em;'><b>⏳ Analyse en cours:</b> {st.session_state.current_question}</div>", unsafe_allow_html=True)
    for entry in qa_history:
        q = entry['question']
        r = entry['result']
        ok = entry.get('success', False)
        method = entry.get('method', '')
        t = entry.get('time', 0)
        cache = entry.get('from_cache', False)
        # Question bubble
        st.markdown(f"<div style='background:#e3e7f1;padding:0.7em 1em;border-radius:10px;margin-bottom:0.2em;'><b>👤 {st.session_state.username}:</b> {q}</div>", unsafe_allow_html=True)
        # Answer bubble
        if ok:
            st.markdown(f"<div style='background:#f6fff6;padding:0.7em 1em;border-radius:10px;margin-bottom:0.5em;'><b>🤖 Agent:</b> ✅ <i>{method}{' - cache' if cache else ''}</i> <small>{t:.2f}s</small>", unsafe_allow_html=True)
            res = r.get('result', None)
            if res is not None:
                # Show result (text, metric, or chart)
                if isinstance(res, (int, float, str)):
                    st.markdown(f"<b>Résultat:</b> {res}", unsafe_allow_html=True)
                elif isinstance(res, pd.Series):
                    if hasattr(res.index, 'dtype') and str(res.index.dtype).startswith('period'):
                        res.index = res.index.astype(str)
                    # Show graph if numeric
                    if res.dtype in ['int64', 'float64'] and len(res) > 0:
                        fig = px.bar(x=res.index, y=res.values, title=f"Analyse: {q[:50]}...", labels={'x': 'Catégorie', 'y': 'Valeur'})
                        fig.update_layout(template="plotly_white", showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(res.to_frame().T, use_container_width=True)
                elif isinstance(res, pd.DataFrame):
                    st.dataframe(res.head(20), use_container_width=True)
                    numeric_cols = res.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fig = px.bar(res.head(10), x=res.index[:10], y=numeric_cols[0], title=f"Visualisation: {q[:50]}...")
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                elif isinstance(res, dict):
                    for k, v in res.items():
                        st.write(f"**{k.title().replace('_', ' ')}:** {v}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#fff6f6;padding:0.7em 1em;border-radius:10px;margin-bottom:0.5em;'><b>🤖 Agent:</b> ❌ <i>{method}</i> <small>{t:.2f}s</small><br><span style='color:red'><b>Erreur:</b> {r.get('error','')}</span></div>", unsafe_allow_html=True)

# --- Input box at the bottom (like a chat app) ---
# --- Input box at the bottom (like a chat app) ---
if st.session_state.username:
    with st.form(key="chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([8,1])
        with col1:
            question = st.text_input("Votre question", value="", placeholder="Tapez votre question ici...", label_visibility="collapsed")
        with col2:
            submit = st.form_submit_button("Envoyer")

    if submit and question:
        qkey = question.strip().lower()
        st.session_state.current_question = question
        # Check cache first
        if qkey in qa_cache:
            result = qa_cache[qkey]
            from_cache = True
        else:
            from_cache = False
            with st.spinner("🔄 Analyse en cours..."):
                t0 = time.time()
                result = agent.intelligent_query_processing(question)
                t1 = time.time()
            # Save to cache if success
            if result['success']:
                qa_cache[qkey] = result
        # Add to history
        qa_history.append({
            'question': question,
            'result': result,
            'success': result['success'],
            'method': result.get('method', ''),
            'time': t1-t0 if not from_cache else 0,
            'from_cache': from_cache
        })
        # Persist for this user (file persistence)
        save_history(st.session_state.username, qa_history)
        save_cache(st.session_state.username, qa_cache)
        st.session_state.qa_history = qa_history
        st.session_state.qa_cache = qa_cache
        st.session_state.current_question = ''
        st.rerun()


# --- Sidebar: Templates and Data Preview ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("📋 Templates Disponibles")
    templates = agent.analysis_templates
    for name, info in templates.items():
        status_icon = "✅" if info['validation'](agent.df) else "❌"
        viz_icon = "📊" if info.get('visualization', False) else "📝"
        with st.expander(f"{status_icon} {viz_icon} {name.replace('_', ' ').title()}"):
            st.write("**Mots-clés:**")
            st.write(", ".join(info['keywords'][:3]))
            if info.get('visualization'):
                st.write("🎨 **Avec visualisation**")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("🔍 Aperçu des Données")
    key_cols = ['NOM_NAVIRE', 'TYPE_NAVIRE', 'PAVILLON']
    available_key_cols = [col for col in key_cols if col in agent.df.columns]
    if available_key_cols:
        sample_df = agent.df[available_key_cols].head(5)
        st.dataframe(sample_df, use_container_width=True)
    else:
        st.warning("Colonnes clés non disponibles")
    st.markdown('</div>', unsafe_allow_html=True)
