"""Streamlit dashboard for monitoring RAG system performance."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from pathlib import Path

from rag_system import SelfImprovingRAG


# ─────────────────────────── Page config ────────────────────────────
st.set_page_config(
    page_title="Self-Improving RAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea18 0%, #764ba218 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda, #b8e6c1);
        border-left: 4px solid #28a745;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffe8a1);
        border-left: 4px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da, #f5b8bd);
        border-left: 4px solid #dc3545;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .chat-bubble-ai {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .ragas-score-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 8px;
        margin: 3px 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Init helpers ───────────────────────────
@st.cache_resource
def load_rag_system():
    """Load RAG system (cached across sessions)."""
    return SelfImprovingRAG()


def _confidence_color(score: float) -> str:
    if score >= 0.75:
        return "#28a745"
    elif score >= 0.6:
        return "#ffc107"
    return "#dc3545"


def _ragas_color(score: float) -> str:
    if score >= 0.75:
        return "#28a745"
    elif score >= 0.5:
        return "#fd7e14"
    return "#dc3545"


# ─────────────────────────── Chart helpers ──────────────────────────
def plot_confidence_trend(feedback_list):
    if not feedback_list:
        return None
    df = pd.DataFrame([
        {"timestamp": f.timestamp, "confidence": f.confidence_score, "is_failure": f.is_failure}
        for f in feedback_list
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["confidence"],
        mode="lines+markers",
        name="Confidence Score",
        line=dict(color="#667eea", width=2.5, shape="spline"),
        marker=dict(
            size=8,
            color=df["is_failure"].map({True: "#dc3545", False: "#28a745"}),
            line=dict(color="white", width=1.5)
        ),
        hovertemplate="Query #%{x}<br>Confidence: %{y:.2%}<extra></extra>"
    ))
    fig.add_hline(
        y=0.6, line_dash="dash", line_color="rgba(220,53,69,0.5)",
        annotation_text="Failure Threshold (0.60)"
    )
    fig.add_hline(
        y=0.75, line_dash="dot", line_color="rgba(40,167,69,0.4)",
        annotation_text="Good Threshold (0.75)"
    )
    fig.update_layout(
        title="Confidence Score Trend",
        xaxis_title="Query #",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        height=380,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,0.8)",
    )
    return fig


def plot_retrieval_performance(optimizer):
    report = optimizer.get_optimization_report()
    methods = report.get("retrieval_methods", {})
    if not methods:
        return None
    data = [
        {"Method": m.capitalize(), "Avg Confidence": s.get("avg_confidence", 0), "Queries": s.get("queries", 0)}
        for m, s in methods.items()
    ]
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="Method", y="Avg Confidence",
        color="Avg Confidence", color_continuous_scale="Viridis",
        text="Queries", title="Retrieval Method Performance"
    )
    fig.update_traces(texttemplate="%{text} queries", textposition="outside", marker_line_width=0)
    fig.update_layout(height=380, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_ragas_radar(scores: dict):
    """Radar/spider chart of RAGAS metric scores."""
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    labels = list(metric_labels.values())
    values = [scores.get(k, 0) for k in metric_labels.keys()]
    values_closed = values + [values[0]]  # Close the polygon
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.2)",
        line=dict(color="#667eea", width=2),
        name="RAGAS Scores",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        showlegend=False,
        height=380,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_ragas_history(ragas_history: list):
    """Line chart showing RAGAS score evolution."""
    if not ragas_history:
        return None
    records = []
    for i, e in enumerate(ragas_history):
        records.append({
            "Index": i + 1,
            "Overall": e.get("ragas_score", 0),
            **e.get("scores", {})
        })
    df = pd.DataFrame(records)
    fig = go.Figure()
    colors = {"Overall": "#667eea", "faithfulness": "#28a745",
              "answer_relevancy": "#764ba2", "context_precision": "#fd7e14", "context_recall": "#17a2b8"}
    for col, color in colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Index"], y=df[col],
                name=col.replace("_", " ").title(),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray", annotation_text="Target 0.70")
    fig.update_layout(
        title="RAGAS Score History",
        xaxis_title="Evaluation #",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        height=380,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─────────────────────────── Sidebar ────────────────────────────────
def render_sidebar(rag):
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.success("✓ System loaded")

        # System info pills
        reranker_status = "✅ On" if rag.reranker.enabled else "⛔ Off"
        ragas_status = "✅ On" if rag.ragas_evaluator.enabled else "⛔ Off"
        st.markdown(f"""
        | Component | Status |
        |---|---|
        | Cross-Encoder Reranker | {reranker_status} |
        | RAGAS Evaluator | {ragas_status} |
        | Hybrid Search | ✅ On |
        """)

        st.divider()
        st.markdown("## 🎯 Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        with col2:
            if st.button("📊 Report", use_container_width=True):
                st.session_state.show_report = True

        st.divider()
        st.markdown("## 📁 Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files",
        )

        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            if st.button("🚀 Ingest Files", type="primary", use_container_width=True):
                total_chunks = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)

                for idx, uf in enumerate(uploaded_files):
                    try:
                        temp_path = temp_dir / uf.name
                        with open(temp_path, "wb") as f:
                            f.write(uf.getbuffer())
                        status_text.text(f"Processing: {uf.name}...")
                        total_chunks += rag.ingest_document(str(temp_path))
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        temp_path.unlink()
                    except Exception as e:
                        st.error(f"Error: {uf.name}: {e}")

                try:
                    temp_dir.rmdir()
                except Exception:
                    pass

                status_text.empty()
                progress_bar.empty()
                st.success(f"✅ Ingested {total_chunks} chunks from {len(uploaded_files)} file(s)!")
                st.balloons()

        st.divider()
        with st.expander("📂 Ingest from Directory"):
            doc_path = st.text_input("Directory Path", placeholder="./documents")
            if st.button("Ingest Directory", use_container_width=True):
                if doc_path and Path(doc_path).exists():
                    try:
                        with st.spinner(f"Ingesting {doc_path}..."):
                            count = rag.ingest_directory(doc_path)
                        st.success(f"✓ Ingested {count} chunks")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Invalid path")


# ─────────────────────────── TAB 1: Overview ─────────────────────────
def render_overview(rag):
    st.header("System Overview")
    stats = rag.get_statistics()
    conf_stats = stats["confidence"]
    feedback_stats = stats["feedback"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Chunks in Store", stats["vector_store"]["total_chunks"])
    with col2:
        st.metric(
            "🎯 Avg Confidence",
            f"{conf_stats['current_avg']:.1%}",
            delta=conf_stats["trend"],
        )
    with col3:
        st.metric("💬 Total Queries", feedback_stats["total_queries"])
    with col4:
        st.metric(
            "❌ Failure Rate",
            f"{feedback_stats['failure_rate']:.1%}",
            delta=f"-{feedback_stats.get('recent_failures', 0)} recent",
            delta_color="inverse",
        )

    st.divider()
    st.subheader("📈 Confidence Trend")
    recent_feedback = rag.feedback_store.get_recent_feedback(100)
    if recent_feedback:
        st.plotly_chart(plot_confidence_trend(recent_feedback), use_container_width=True)
    else:
        st.info("No query history yet. Start querying to see trends!")

    st.subheader("🏥 System Health")
    avg = conf_stats["current_avg"]
    if avg >= 0.75:
        st.markdown('<div class="success-box">✅ <b>System is performing well!</b> Average confidence ≥ 75%.</div>', unsafe_allow_html=True)
    elif avg >= 0.6:
        st.markdown('<div class="warning-box">⚠️ <b>Moderate performance.</b> Consider running optimization.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ <b>Poor performance.</b> Optimization needed urgently!</div>', unsafe_allow_html=True)


# ─────────────────────────── TAB 2: Streaming Query ──────────────────
def render_query_interface(rag):
    st.header("💬 Query Interface")

    # Initialize session state keys
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "auto_submit" not in st.session_state:
        st.session_state.auto_submit = False

    # Settings row
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        top_k = st.slider("Documents to retrieve", 1, 10, 5, key="query_top_k")
    with col_s2:
        use_streaming = st.toggle("⚡ Streaming", value=True, help="Stream the answer token-by-token")
    with col_s3:
        show_metadata = st.toggle("🔍 Show metadata", value=True)

    # ── Example questions ─────────────────────────────────────────
    st.markdown("##### 💡 Example Questions")
    examples = [
        "What are the main topics covered in the documents?",
        "Summarize the key findings.",
        "What methods or approaches are described?",
        "What conclusions are drawn?",
    ]
    example_cols = st.columns(2)
    for idx, ex in enumerate(examples):
        with example_cols[idx % 2]:
            if st.button(ex, key=f"ex_{idx}", use_container_width=True):
                # Write directly into the text_area's widget state key
                # AND set auto_submit so we fire the query immediately
                st.session_state["query_input"] = ex
                st.session_state.auto_submit = True
                st.rerun()

    st.divider()

    # ── Query text area ───────────────────────────────────────────
    # Initialise the widget key if it doesn't exist yet
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""

    query = st.text_area(
        "✏️ Or type your own question:",
        height=90,
        placeholder="Ask anything about your documents...",
        key="query_input",
    )

    submit_col, clear_col = st.columns([3, 1])
    with submit_col:
        submit = st.button("🔍 Submit Query", type="primary", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_metadata = None
            st.rerun()

    # Trigger from either manual Submit click OR auto_submit flag (example click)
    auto_submit = st.session_state.get("auto_submit", False)
    if auto_submit:
        st.session_state["auto_submit"] = False
    if (submit or auto_submit) and query.strip():
        if use_streaming:
            # ── Streaming path ────────────────────────────────────
            pipeline_placeholder = st.empty()
            answer_placeholder = st.empty()
            accumulated = ""
            final_meta = None

            gen = rag.stream_query(query.strip(), top_k=top_k)
            for event in gen:
                if event["type"] == "status":
                    # Don't overwrite the streamed answer with the RAGAS status line
                    if accumulated:
                        pass  # answer already streaming — ignore status updates
                    else:
                        pipeline_placeholder.info(event["text"])
                elif event["type"] == "token":
                    accumulated += event["text"]
                    pipeline_placeholder.empty()
                    answer_placeholder.markdown(accumulated + "▍")  # blinking cursor feel
                elif event["type"] == "done":
                    final_meta = event["metadata"]

            # Final render — no cursor
            answer_placeholder.markdown(accumulated)

            if final_meta:
                conf = final_meta["confidence_score"]
                ragas = final_meta.get("ragas")

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Confidence", f"{conf:.1%}")
                if ragas:
                    m2.metric("📐 RAGAS Score", f"{ragas.get('ragas_score', 0):.1%}")
                    scores = ragas.get("scores", {})
                    m3.metric("🛡️ Faithfulness", f"{scores.get('faithfulness', 0):.1%}")

                # Store in history
                st.session_state.chat_history.append({
                    "query": query.strip(),
                    "answer": accumulated,
                    "confidence": conf,
                    "reranked": True,
                    "metadata": final_meta,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                st.session_state.last_metadata = final_meta

                if show_metadata:
                    with st.expander("📄 Retrieved Documents"):
                        for i, chunk in enumerate(final_meta.get("retrieved_chunks", [])):
                            score_str = ""
                            if "rerank_score" in chunk:
                                score_str = f" | Rerank score: {chunk['rerank_score']:.3f}"
                            elif "combined_score" in chunk:
                                score_str = f" | Hybrid score: {chunk['combined_score']:.3f}"
                            st.markdown(f"**Document {i+1}**{score_str}")
                            st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                            st.divider()

                    with st.expander("🔬 Evaluation Details"):
                        st.json(final_meta.get("evaluation", {}))

                    if ragas:
                        with st.expander("📐 RAGAS Breakdown"):
                            scores = ragas.get("scores", {})
                            rc1, rc2, rc3, rc4 = st.columns(4)
                            rc1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.0%}")
                            rc2.metric("Relevancy", f"{scores.get('answer_relevancy', 0):.0%}")
                            rc3.metric("Ctx Precision", f"{scores.get('context_precision', 0):.0%}")
                            rc4.metric("Ctx Recall", f"{scores.get('context_recall', 0):.0%}")
        else:
            # ── Non-streaming path ────────────────────────────────
            with st.spinner("Processing query..."):
                try:
                    result = rag.query(query.strip(), top_k=top_k, return_metadata=show_metadata)

                    st.subheader("Answer")
                    st.markdown(result["answer"])

                    conf = result["confidence_score"]
                    st.metric("Confidence Score", f"{conf:.1%}")

                    st.session_state.chat_history.append({
                        "query": query.strip(),
                        "answer": result["answer"],
                        "confidence": conf,
                        "reranked": result.get("reranked", False),
                        "metadata": result if show_metadata else {},
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })

                    if show_metadata:
                        if result.get("rewritten_query") and result["rewritten_query"] != query:
                            st.info(f"**Query Rewritten:** {result['rewritten_query']}")
                        with st.expander("📄 Retrieved Documents"):
                            for i, chunk in enumerate(result.get("retrieved_chunks", [])):
                                st.markdown(f"**Document {i+1}**")
                                st.text(chunk["text"][:400] + "...")
                                if "rerank_score" in chunk:
                                    st.caption(f"Rerank Score: {chunk['rerank_score']:.3f}")
                                elif "combined_score" in chunk:
                                    st.caption(f"Hybrid Score: {chunk['combined_score']:.3f}")
                                st.divider()

                        if result.get("ragas"):
                            with st.expander("📐 RAGAS Scores"):
                                ragas = result["ragas"]
                                rc1, rc2, rc3, rc4 = st.columns(4)
                                scores = ragas.get("scores", {})
                                rc1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.0%}")
                                rc2.metric("Relevancy", f"{scores.get('answer_relevancy', 0):.0%}")
                                rc3.metric("Ctx Precision", f"{scores.get('context_precision', 0):.0%}")
                                rc4.metric("Ctx Recall", f"{scores.get('context_recall', 0):.0%}")

                except Exception as e:
                    st.error(f"Error processing query: {e}")

    elif submit:
        st.warning("Please enter a question.")


# ─────────────────────────── TAB 3: Chat History ─────────────────────
def render_chat_history():
    st.header("📜 Chat History")

    history = st.session_state.get("chat_history", [])

    if not history:
        st.info("No queries yet — go to the **Query Interface** tab to start chatting!")
        return

    st.markdown(f"**{len(history)} conversation(s) this session**")

    # Filter bar
    search = st.text_input("🔎 Search history", placeholder="Filter by keyword...")

    # Reverse to show newest first
    filtered = [h for h in reversed(history) if not search or search.lower() in h["query"].lower() or search.lower() in h["answer"].lower()]

    if not filtered:
        st.warning("No results match your search.")
        return

    for i, entry in enumerate(filtered):
        conf = entry["confidence"]
        color = _confidence_color(conf)
        badge = f'<span style="background:{color}22; color:{color}; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">{conf:.0%}</span>'

        reranked_badge = ""
        if entry.get("reranked"):
            reranked_badge = '<span style="background:#667eea22; color:#667eea; padding:2px 8px; border-radius:12px; font-size:0.75rem;">⚡ reranked</span>'

        with st.expander(
            f"**Q{len(history)-i}** [{entry['timestamp']}] {entry['query'][:70]}...",
            expanded=(i == 0)
        ):
            st.markdown(
                f'<div class="chat-bubble-user">🧑 <b>You</b> &nbsp; {badge} {reranked_badge}<br><br>{entry["query"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chat-bubble-ai">🤖 <b>Assistant</b><br><br>{entry["answer"]}</div>',
                unsafe_allow_html=True
            )

            meta = entry.get("metadata", {})
            if meta:
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    rw = meta.get("rewritten_query", "")
                    if rw and rw != entry["query"]:
                        st.caption(f"🔁 Rewritten: _{rw}_")
                with sub_col2:
                    method = meta.get("retrieval_method", "")
                    if method:
                        st.caption(f"📡 Retrieval: **{method}**")

    # Export button
    st.divider()
    if st.button("📥 Export History as CSV", use_container_width=True):
        df = pd.DataFrame([
            {
                "timestamp": h["timestamp"],
                "query": h["query"],
                "answer": h["answer"],
                "confidence": h["confidence"],
            }
            for h in history
        ])
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────── TAB 4: Analytics ────────────────────────
def render_analytics(rag):
    st.header("📈 Analytics")

    st.subheader("🎯 Retrieval Method Performance")
    fig = plot_retrieval_performance(rag.optimizer)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No retrieval data yet.")

    st.divider()

    st.subheader("❌ Recent Failures")
    failures = rag.feedback_store.get_failure_cases(10)
    if failures:
        for i, failure in enumerate(failures[-5:]):
            with st.expander(f"Failure {i+1}: {failure.query[:60]}..."):
                st.write(f"**Query:** {failure.query}")
                st.write(f"**Answer:** {failure.answer}")
                st.write(f"**Confidence:** {failure.confidence_score:.1%}")
                st.write(f"**Reason:** {failure.failure_reason}")
    else:
        st.success("No failures recorded! 🎉")


# ─────────────────────────── TAB 5: RAGAS ────────────────────────────
def render_ragas(rag):
    st.header("📐 RAGAS Evaluation Metrics")
    st.markdown("Industry-standard retrieval-augmented generation quality metrics evaluated via LLM-as-judge.")

    evaluator = rag.ragas_evaluator

    if not evaluator.enabled:
        st.warning("RAGAS evaluator is disabled. Enable it in `config.yaml` under `evaluation.ragas_enabled: true`.")
        return

    history = evaluator.get_history(last_n=50)
    agg = evaluator.get_aggregate_stats()

    # ── Aggregate metrics ─────────────────────────────────────────
    st.subheader("📊 Aggregate Scores")
    overall = agg.get("overall_ragas", 0.0)
    color = _ragas_color(overall)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "🏆 RAGAS Score",
        f"{overall:.1%}",
        help="Average of all 4 RAGAS metrics"
    )
    col2.metric(
        "🛡️ Faithfulness",
        f"{agg.get('faithfulness', {}).get('mean', 0):.1%}",
        help="Fraction of answer claims grounded in retrieved context"
    )
    col3.metric(
        "🎯 Answer Relevancy",
        f"{agg.get('answer_relevancy', {}).get('mean', 0):.1%}",
        help="How well the answer addresses the question"
    )
    col4.metric(
        "🎁 Context Precision",
        f"{agg.get('context_precision', {}).get('mean', 0):.1%}",
        help="Fraction of retrieved chunks that are actually relevant"
    )
    col5.metric(
        "📡 Context Recall",
        f"{agg.get('context_recall', {}).get('mean', 0):.1%}",
        help="Whether context contains all info needed to answer"
    )

    st.divider()

    if not history:
        st.info("No RAGAS evaluations yet — run some queries first!")
        return

    # ── Last evaluation radar ─────────────────────────────────────
    col_radar, col_last = st.columns([1, 1])

    with col_radar:
        st.subheader("🕸️ Latest Evaluation")
        last_eval = history[-1]
        radar_fig = plot_ragas_radar(last_eval.get("scores", {}))
        st.plotly_chart(radar_fig, use_container_width=True)

    with col_last:
        st.subheader("🔬 Latest Query Details")
        last = history[-1]
        st.markdown(f"**Query:** {last.get('query', 'N/A')}")
        st.markdown(f"**Overall RAGAS:** `{last.get('ragas_score', 0):.1%}`")

        scores = last.get("scores", {})
        for metric, label in [
            ("faithfulness", "🛡️ Faithfulness"),
            ("answer_relevancy", "🎯 Answer Relevancy"),
            ("context_precision", "🎁 Context Precision"),
            ("context_recall", "📡 Context Recall"),
        ]:
            val = scores.get(metric, 0)
            c = _ragas_color(val)
            st.markdown(
                f'{label}: <b style="color:{c}">{val:.0%}</b>',
                unsafe_allow_html=True
            )

        details = last.get("details", {})
        if details:
            with st.expander("📝 Detailed Explanations"):
                for metric, data in details.items():
                    if isinstance(data, dict) and "reasoning" in data:
                        st.markdown(f"**{metric.replace('_', ' ').title()}**: {data['reasoning']}")

    st.divider()

    # ── History chart ──────────────────────────────────────────────
    st.subheader("📈 RAGAS Score History")
    history_fig = plot_ragas_history(history)
    if history_fig:
        st.plotly_chart(history_fig, use_container_width=True)

    # ── History table ──────────────────────────────────────────────
    st.subheader("📋 Evaluation Log")
    table_data = []
    for i, e in enumerate(reversed(history[-20:])):
        s = e.get("scores", {})
        table_data.append({
            "#": len(history) - i,
            "Query": e.get("query", "")[:60] + "...",
            "RAGAS": f"{e.get('ragas_score', 0):.0%}",
            "Faithfulness": f"{s.get('faithfulness', 0):.0%}",
            "Relevancy": f"{s.get('answer_relevancy', 0):.0%}",
            "Precision": f"{s.get('context_precision', 0):.0%}",
            "Recall": f"{s.get('context_recall', 0):.0%}",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Export RAGAS history
    if st.button("📥 Export RAGAS History", use_container_width=True):
        export_data = []
        for e in history:
            s = e.get("scores", {})
            export_data.append({
                "query": e.get("query", ""),
                "ragas_score": e.get("ragas_score", 0),
                **s,
            })
        df = pd.DataFrame(export_data)
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            file_name=f"ragas_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────── TAB 6: Optimization ─────────────────────
def render_optimization(rag):
    st.header("🔧 System Optimization")

    opt_report = rag.optimizer.get_optimization_report()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Retrieval Method", opt_report["best_retrieval_method"].capitalize())
    with col2:
        st.metric("Problematic Documents", len(opt_report["problematic_documents"]))

    if opt_report["problematic_documents"]:
        st.subheader("⚠️ Problematic Documents")
        st.write("These documents consistently produce low-confidence answers:")
        for doc_id in opt_report["problematic_documents"][:10]:
            st.code(doc_id)
        if st.button("🔄 Trigger Re-chunking", type="primary"):
            st.info("Re-chunking would be triggered here in production.")

    st.divider()
    st.subheader("🛠️ Manual Actions")

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("🔨 Rebuild BM25 Index", use_container_width=True):
            rag.retriever.rebuild_index()
            st.success("✓ BM25 index rebuilt")
    with action_col2:
        if st.button("🧹 Clear Failure Memory", use_container_width=True):
            rag.feedback_store.failure_memory = []
            st.success("✓ Failure memory cleared")
    with action_col3:
        if st.button("🔁 Reload Cross-Encoder", use_container_width=True):
            rag.reranker._load_model()
            st.success("✓ Cross-encoder reloaded")


# ─────────────────────────── Main app ────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🧠 Self-Improving RAG Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monitor, query, and evaluate your adaptive RAG system in real-time</p>', unsafe_allow_html=True)

    try:
        rag = load_rag_system()
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}")
        return

    render_sidebar(rag)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "💬 Query",
        "📜 Chat History",
        "📈 Analytics",
        "📐 RAGAS",
        "🔧 Optimization",
    ])

    with tab1:
        render_overview(rag)
    with tab2:
        render_query_interface(rag)
    with tab3:
        render_chat_history()
    with tab4:
        render_analytics(rag)
    with tab5:
        render_ragas(rag)
    with tab6:
        render_optimization(rag)


if __name__ == "__main__":
    main()
