import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")
IDENTITIES = {
    "พนักงาน (demo)": "employee-demo-token",
    "ผู้ดูแลระบบ (demo)": "admin-demo-token",
}
STATUS_ICON = {"answered": "✅", "fallback": "⚠️", "refused": "⛔", "error": "❌"}

st.set_page_config(page_title="Mini Enterprise RAG", layout="wide")
st.title("Mini Enterprise RAG — ข้อมูลจำลอง")
st.caption("เอกสารทั้งหมดเป็นข้อมูลสมมติเพื่อการสาธิต ไม่ใช่นโยบายจริงขององค์กรใด")

identity = st.sidebar.selectbox("ตัวตน (demo auth)", list(IDENTITIES))
headers = {"Authorization": f"Bearer {IDENTITIES[identity]}"}


def call(method: str, path: str, **kwargs: Any) -> Any:
    try:
        r = httpx.request(method, f"{API}{path}", headers=headers, timeout=600, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"เชื่อมต่อ API ไม่ได้: {exc}")
        return None
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.json().get('error', {}).get('message', r.text)}")
        return None
    return r.json()


def render_answer(resp: dict[str, Any]) -> None:
    st.markdown(
        f"{STATUS_ICON.get(resp['status'], '')} **{resp['status']}**"
        + (f" · `{resp['reason_code']}`" if resp["reason_code"] else "")
    )
    st.write(resp["answer"])
    if resp["sources"]:
        st.markdown("**แหล่งอ้างอิง**")
        for s in resp["sources"]:
            with st.expander(f"{s['document_id']} v{s['version']} — {s['title']}"):
                st.text(s["snippet"])
    elif resp["decision_type"] == "system_policy":
        st.caption("การปฏิเสธนี้มาจากนโยบายของระบบ ไม่ได้อ้างอิงเอกสาร")

    with st.expander("Debug inspector"):
        d = resp["debug"]
        st.write("**Node sequence:** " + " → ".join(d["trace"]))
        st.write(
            f"**Gate passed:** {d['gate_passed']} · **best similarity:** "
            f"{resp['retrieval']['best_similarity']} · **threshold:** "
            f"{resp['retrieval']['threshold']}"
        )
        st.write(
            f"**Latency:** {d['latency_ms']} ms · **model:** {d['model']} · "
            f"**audit:** {resp['audit_status']} · **request_id:** `{resp['request_id']}`"
        )
        if d["hits"]:
            st.dataframe(pd.DataFrame(d["hits"]), hide_index=True)


chat_tab, docs_tab, eval_tab, audit_tab = st.tabs(["Chat", "Documents", "Evaluation", "Audit"])

with chat_tab:
    st.session_state.setdefault("history", [])
    for question, resp in st.session_state.history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_answer(resp)
    if question := st.chat_input("ถามเรื่องการลา / การเบิกค่าใช้จ่าย เช่น ลาป่วยทำไง"):
        resp = call("POST", "/api/v1/chat", json={"query": question})
        if resp:
            st.session_state.history.append((question, resp))
            st.rerun()

with docs_tab:
    docs = call("GET", "/api/v1/documents")
    if docs:
        st.dataframe(pd.DataFrame(docs), hide_index=True)
    if st.button("Ingest corpus (admin)"):
        report = call("POST", "/api/v1/admin/ingest")
        if report:
            st.json(report)

with eval_tab:
    thresholds = st.multiselect(
        "Thresholds", [0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50], default=[0.25, 0.28, 0.35]
    )
    if st.button("Run evaluation (admin, calls the LLM)") and thresholds:
        with st.spinner("Running 40 cases per threshold..."):
            call("POST", "/api/v1/admin/evaluations/run", json={"thresholds": thresholds})
    runs = call("GET", "/api/v1/admin/evaluations", params={"limit": 5}) or []
    for run in runs:
        st.subheader(f"{run['created_at'][:19]} · dataset {run['dataset_version']}")
        st.caption(
            f"config: {run['config_snapshot']} · recommended threshold: "
            f"{run['metrics']['recommended_threshold']}"
        )
        for r in run["metrics"]["runs"]:
            with st.expander(
                f"threshold {r['threshold']} — accuracy {r['metrics']['behavior_accuracy']}"
            ):
                metrics = {k: v for k, v in r["metrics"].items() if not isinstance(v, dict)}
                st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))
                results = pd.DataFrame(r["results"])
                st.markdown("**Failures**")
                st.dataframe(results[~results["passed"]], hide_index=True)
                st.download_button(
                    "Download CSV",
                    results.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"eval_{run['id']}_{r['threshold']}.csv",
                    key=f"{run['id']}-{r['threshold']}",
                )

with audit_tab:
    logs = call("GET", "/api/v1/admin/audit-logs", params={"limit": 100})
    if logs:
        st.dataframe(pd.DataFrame(logs), hide_index=True)
