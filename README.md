# Mini Enterprise RAG

Interview prototype for an AI Forward Deployed Engineer exercise. It answers Thai HR and expense questions from a **synthetic** employee handbook, cites the document it used, tolerates typos and slang, refuses prompt injection, and says clearly when the handbook has no answer.

> The handbook in `data/` is synthetic demo data. Nothing here is a real company policy.

| | URL | For |
|---|---|---|
| Chat | http://localhost:8000/ | Employees asking questions |
| Admin console | http://localhost:8501/ | Debugging answers, documents, evaluation, audit log |
| API docs | http://localhost:8000/docs | Developers |

## Architecture

![Architecture](docs/images/architecture.svg)

- **Employee** uses the chat page; **admin** uses the Streamlit console. Both call the same FastAPI service.
- FastAPI checks the bearer token, then runs one **LangGraph** pipeline per question.
- The pipeline calls four services: **guardrails** (in-process rules), **local embeddings** (fastembed, no API key), **Claude Opus 5.5** (query rewrite and answer), and **PostgreSQL + pgvector** (search, documents, audit log).
- **Ingest** and **evaluation** are separate jobs that reuse the same code: ingest fills the vector table, evaluation runs the 40 test questions through the same graph.
- Everything runs in Docker Compose: `db`, `api` (also serves the chat page), `ui`.

## Demo

**Typo + question not in the handbook.** "ลาป๊วยทำไง" is rewritten to the handbook wording and answered from HR-001. "ลาคลอดได้กี่วัน" is not covered, so the reply says so and lists what can be asked.

![Chat: typo and missing topic](docs/images/chat-typo-and-missing.png)

**Urgent leave, prompt injection, unreadable text.** The urgent-leave answer is a numbered procedure citing HR-002. The injection attempt is refused in 2 ms without calling the model. Random keystrokes get a request to rephrase.

![Chat: urgent leave, injection, unclear](docs/images/chat-urgent-injection-unclear.png)

**Admin console, Debug inspector.** Node sequence, gate result, top-5 hits with similarity, latency, model and request ID for one question.

![Admin: debug inspector](docs/images/admin-debug-inspector.png)

**Admin console, Documents.** The 11 ingested handbook documents (6 Finance, 5 Human Resources), all `SOP`, `internal`, version 1.

![Admin: documents](docs/images/admin-documents.png)

## วิธีใช้ (How to use)

### 1. เริ่มระบบครั้งแรก

```bash
cd mini-enterprise-rag
cp .env.example .env          # ใส่ ANTHROPIC_API_KEY ในไฟล์ .env
make up                       # build + start db/api/ui + migrate (ครั้งแรกนาน เพราะโหลดโมเดล ~1 GB)
make ingest                   # โหลดคู่มือจาก data/documents.jsonl
```

ถ้าพอร์ต 5432 ถูกใช้อยู่ ใส่ `DB_HOST_PORT=55499` ใน `.env` ก่อน `make up`

### 2. ถามคำถาม (พนักงาน)

1. เปิด http://localhost:8000/
2. พิมพ์คำถามแล้วกด Enter หรือกดคำถามตัวอย่างบนหน้าจอ
3. อ่านป้ายสถานะเหนือคำตอบ
   - **ตอบจากเอกสาร** (เขียว): กด "แหล่งอ้างอิง" เพื่อดูข้อความต้นฉบับ
   - **ข้อมูลไม่พอ** (เหลือง): คู่มือไม่มีเรื่องนี้ หรือระบบไม่เข้าใจคำถาม ระบบจะบอกหัวข้อที่ถามได้
   - **ปฏิเสธ** (แดง): คำขอผิดนโยบาย เช่นสั่งให้เปิดเผยกฎ
4. พิมพ์ผิดหรือใช้ภาษาพูดได้ เช่น "ลาป๊วยทำไง", "ใบเสด็จหายอะ" ถ้าระบบแก้คำให้ จะเห็นบรรทัด `ค้นหาเป็น: "..."`
5. ระบบไม่จำคำถามก่อนหน้า ให้ถามให้ครบในประโยคเดียว

ตัวอย่างคำถามที่ถามได้

| หมวด | ตัวอย่าง |
|---|---|
| การลา | ลาป่วยต้องแนบใบรับรองแพทย์ไหม · มีธุระด่วนต้องลาวันนี้เลย ทำไงดี · วันลาเหลือดูตรงไหน · กดลาผิด ยกเลิกไง |
| การเบิก | เบิกค่าแท็กซี่ไปหาลูกค้าได้ไหม · ค่าอาหารต่างจังหวัดวันละเท่าไหร่ · ใบเสร็จหายทำไง · เงินเบิกเข้าเมื่อไหร่ |

### 3. ตรวจว่าทำไมระบบตอบแบบนั้น (ผู้ดูแล)

1. เปิด http://localhost:8501/ แล้วเลือก **ผู้ดูแลระบบ (demo)** ที่ sidebar
2. แท็บ **Chat** ถามคำถามเดียวกัน แล้วกด **Debug inspector**
3. ดู 3 อย่าง: เอกสารที่ควรเจออยู่ใน 5 อันดับแรกไหม, `best similarity` ผ่าน threshold ไหม, และ node sequence ผ่าน `rewrite_query` หรือเปล่า
4. แท็บ **Audit** ดูประวัติทุกคำถาม (เก็บเป็น hash ไม่เก็บข้อความจริง)

### 4. เปลี่ยนหรือเพิ่มเอกสาร

1. แก้ `data/documents.jsonl` (รูปแบบอยู่ในหัวข้อ [Updating the handbook](#updating-the-handbook))
2. รัน `make reset-docs` ระบบตรวจไฟล์ก่อน ถ้าผิดจะหยุดโดยไม่ลบข้อมูลเดิม
3. เปิดแท็บ **Documents** ใน http://localhost:8501/ เพื่อยืนยัน แล้วลองถามที่ http://localhost:8000/
4. ถ้าเนื้อหาเปลี่ยน ให้ปรับ `data/evaluation.jsonl` ให้ตรง แล้วรัน eval ข้อ 5

### 5. วัดผลหลังแก้ระบบ

1. รัน `make eval` หรือแท็บ **Evaluation** ใน http://localhost:8501/ แล้วกด **Run evaluation** (เรียก Claude จริง มีค่าใช้จ่าย)
2. ดู `behavior_accuracy`, `false_answer_rate`, `unnecessary_fallback_rate` และตาราง **Failures**
3. ผลเก็บใน `reports/eval_*.json` และดาวน์โหลด CSV จากหน้า Evaluation ได้

### 6. หยุดระบบ

```bash
make down                     # หยุด container เก็บข้อมูลไว้
docker compose down -v        # หยุดและลบฐานข้อมูลทั้งหมด (ย้อนกลับไม่ได้)
```

### 7. คำสั่ง Makefile

พิมพ์ `make` หรือ `make help` เพื่อดูรายการคำสั่งทั้งหมด ทุกคำสั่งรันจากโฟลเดอร์ `mini-enterprise-rag/`

| คำสั่ง | ทำอะไร | ใช้เมื่อ |
|---|---|---|
| `make up` | build image, เปิด db / api / ui, migrate DB | เริ่มระบบ หรือหลังแก้โค้ด |
| `make ingest` | โหลด `data/documents.jsonl` เข้า DB (รันซ้ำได้ ไม่ซ้ำข้อมูล) | ครั้งแรก หรือเพิ่มเอกสารใหม่ |
| `make reset-docs` | ตรวจไฟล์ก่อน ถ้าผ่านค่อยลบเอกสารเดิมทั้งหมดแล้วโหลดใหม่ | เปลี่ยนชุดเอกสารทั้งชุด |
| `make eval` | รันคำถามทดสอบ 40 ข้อที่ threshold 0.25 / 0.28 / 0.35 (เรียก Claude) | หลังแก้เอกสาร prompt หรือโค้ด |
| `make down` | หยุด container ข้อมูลยังอยู่ | เลิกใช้งาน |
| `make migrate` | อัปเดต schema ของ DB | หลังดึงโค้ดใหม่ที่มี migration |
| `make setup` | สร้าง `.venv` ลง dependency สำหรับพัฒนา | ครั้งแรกบนเครื่องที่จะแก้โค้ด |
| `make check` | ruff + mypy + unit test 78 ข้อ (ไม่ต้องมี DB หรือเน็ต) | ก่อน commit |
| `make test-integration` | เทสต์ pgvector ใน DB แยกชื่อ `rag_test` ไม่แตะข้อมูลจริง | หลังแก้ส่วน DB หรือ ingest |
| `make test-live` | เทสต์ที่เรียก provider จริง (มีค่าใช้จ่าย) | ตรวจ key หรือ provider |
| `make format` | แก้ format และ lint อัตโนมัติ | ก่อน `make check` |

ลำดับที่ใช้บ่อย

```bash
make up && make ingest        # เปิดระบบครั้งแรก
make reset-docs               # เปลี่ยนชุดเอกสาร
make eval                     # วัดผลหลังแก้
make down                     # ปิด
```

ตั้งค่าเพิ่มผ่านตัวแปร เช่น `make setup PYTHON=python3.12` หรือใส่ `DB_HOST_PORT=55499` ใน `.env` เมื่อพอร์ต 5432 ถูกใช้อยู่ (Makefile อ่านค่านี้จาก `.env` เอง)

### 8. ย้ายไปรันเครื่องอื่น

**เครื่องปลายทางต้องมี**

- Docker Desktop (macOS / Windows) หรือ Docker Engine + Compose v2 (Linux) บน Windows ให้รันผ่าน WSL2
- `make` (macOS มีใน Xcode Command Line Tools, Linux ติดตั้ง `build-essential`)
- พื้นที่ดิสก์ว่างอย่างน้อย 6 GB (image ~3.4 GB + PostgreSQL)
- อินเทอร์เน็ตตอน build ครั้งแรก (โหลด Python packages และโมเดล embedding ~1 GB) หลังจากนั้นเรียกเน็ตแค่ตอนคุยกับ Claude
- `ANTHROPIC_API_KEY`
- ไม่ต้องติดตั้ง Python ถ้าแค่รันระบบ (ต้องมีเฉพาะถ้าจะใช้ `make setup` / `make check`)

**ขั้นตอน**

```bash
# 1. คัดลอกโปรเจกต์ (ไม่ต้องเอา .venv, .cache และห้ามเอา .env ที่มี key ติดไป)
rsync -a --exclude .venv --exclude .cache --exclude .env --exclude '__pycache__' \
  mini-enterprise-rag/ user@other-machine:~/mini-enterprise-rag/
#    หรือ git clone / zip ก็ได้ ไฟล์ .env ถูก .gitignore ไว้แล้ว

# 2. บนเครื่องปลายทาง
cd ~/mini-enterprise-rag
cp .env.example .env          # ใส่ ANTHROPIC_API_KEY และเปลี่ยน AUDIT_HASH_SALT เป็นค่าสุ่มยาวๆ
make up                       # build ครั้งแรก 5-15 นาที
make ingest

# 3. เปิดใช้งาน
#    http://localhost:8000/      หน้า chat
#    http://localhost:8501/      หน้า admin
```

**ตรวจว่าใช้ได้**

```bash
curl -s localhost:8000/health/ready          # ควรได้ {"database":"ok","provider":"anthropic",...}
make eval                                    # ควรได้ behavior_accuracy ใกล้ 1.0
```

**ปัญหาที่เจอบ่อย**

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `port is already allocated` ตอน `make up` | มีโปรแกรมอื่นใช้พอร์ต 5432 ใส่ `DB_HOST_PORT=55499` ใน `.env` แล้ว `make up` ใหม่ ถ้าเป็นพอร์ต 8000 หรือ 8501 ต้องปิดโปรแกรมที่ใช้พอร์ตนั้น |
| build ค้างหรือ Docker ล่ม | ดิสก์เต็ม ดูด้วย `docker system df` และ `df -h` |
| ตอบทุกข้อว่า "ระบบขัดข้อง" (`PROVIDER_UNAVAILABLE`) | key ผิดหรือเครดิตหมด ดู `docker compose logs api` |
| ตอบทุกข้อว่า "ไม่มีข้อมูล" | ยังไม่ได้ `make ingest` ตรวจที่แท็บ Documents ใน :8501 |
| api ไม่ขึ้น, log มี `outputs 768 dimensions, EMBEDDING_DIMENSIONS=...` | `EMBEDDING_DIMENSIONS` ใน `.env` ไม่ใช่ 768 แก้แล้วรัน `docker compose down -v && make up && make ingest` |
| Mac ชิป Apple / Linux x86 | ใช้ได้ทั้งคู่ image build ตามสถาปัตยกรรมของเครื่องเอง |

## Results

Latest evaluation: 40 labelled cases against the current 11-document handbook (dataset `25693c9d7cb0`, run 2026-11-22 16:52 UTC, report `reports/eval_20261122T165254.json`).

| Metric | t = 0.25 | **t = 0.28 (in use)** | t = 0.35 |
|---|---|---|---|
| Behaviour accuracy | 1.00 | **1.00** | 1.00 |
| Recall@5 | 1.00 | **1.00** | 1.00 |
| MRR | 0.95 | **0.95** | 0.95 |
| Citation precision | 1.00 | **0.95** | 1.00 |
| Fact match (lexical) | 1.00 | **1.00** | 1.00 |
| Injection refusal recall | 1.00 | **1.00** | 1.00 |
| False answer rate | 0.00 | **0.00** | 0.00 |
| Unnecessary fallback rate | 0.00 | **0.00** | 0.00 |
| Latency p50 / p95 | 3.1 s / 5.3 s | **2.9 s / 4.9 s** | 2.9 s / 4.6 s |

All six categories passed at every threshold: SOP 10/10, slang and typos 10/10, ambiguous 5/5, out of scope 5/5, injection 5/5, missing knowledge 5/5.

![Admin: evaluation run, threshold 0.28, no failures](docs/images/admin-evaluation.png)

Run history, as exported from the admin console in [`docs/eval-results-2026-11-22.pdf`](docs/eval-results-2026-11-22.pdf):

| Run (UTC) | Dataset | Accuracy | Note |
|---|---|---|---|
| 16:30 | `57a95b7a81ec` | 1.00 | First eval set for the new handbook, t = 0.28 only |
| 16:40, 16:41 | `57a95b7a81ec` | 0.975 | SLANG-09 "มีธุระด่วนต้องลาวันนี้เลย" failed: the answer used a numbered list and the number check flagged "1." and "2." as invented facts. Fixed in `output_policy.py` |
| **16:52** | **`25693c9d7cb0`** | **1.00** | **Eval set rewritten for the current handbook content (the results above)** |
| 17:03 | `57a95b7a81ec` | 1.00 | Run from the admin console while the API image still held the previous eval file. `data/` is now mounted into the container, so this cannot happen again |

<details>
<summary>PDF export of the Evaluation tab (2 pages)</summary>

![Evaluation export, page 1](docs/images/eval-results-page-1.png)
![Evaluation export, page 2](docs/images/eval-results-page-2.png)

</details>

**How to read this.** The cases were written after seeing the documents and the system, so they work as a regression suite, not as an accuracy estimate. All three thresholds score the same because questions with no answer are stopped by Claude (`MODEL_CANNOT_ANSWER`), not by the threshold. A held-out set written by someone else is needed before quoting a quality number. Fact match is a substring check, and citation precision moves a little between runs because Claude sometimes cites one extra related document.

## How a question flows

![How a question flows](docs/images/question-flow.svg)

- **Green**: normal path. **Pink dashed**: one retry with a rewritten query. **Orange**: fallback reasons. Refusals and fallbacks are logged like answers.

| Node | What it does |
|---|---|
| `validate_input` | Length check, Unicode cleanup, regex rules for injection and out-of-scope topics, PII redaction before anything leaves the service |
| `retrieve` | Embeds the query and runs cosine top-5 over chunks the caller is allowed to see |
| `evidence_gate` | Best similarity must reach the threshold and at least one SOP chunk must be present |
| `rewrite_query` | Runs at most once, only when evidence is weak or Claude could not answer. Claude rewrites typos and slang into the handbook's own wording (it sees the document titles), e.g. "ลาพักล้อนเหลือกี่วันดูไง" becomes "ตรวจสอบวันลาพักร้อนคงเหลือ". If the text is unreadable it returns `UNCLEAR_QUERY` |
| `generate` | Claude with structured output (`answer`, `cited_source_ids`, `cannot_answer`, `needs_clarification`). Sources are labelled `S1..Sn` and mapped back to chunk IDs on the server, so the model cannot invent an ID |
| `verify_output` | Citations must be retrieved chunks. Every number in the answer must appear in the cited text (list numbering such as "1." is ignored) |
| `fallback` | Friendly Thai message, `sources: []`, and a list of topics the user can ask about, built from the documents they are allowed to see |
| `audit_and_finalize` | Writes `query_logs` and `retrieval_hits` with HMAC-hashed actor and query. If the audit write fails the answer is still returned with `audit_status: "degraded"` |

### What the user sees

| Situation | Reply |
|---|---|
| Answer found | Answer starting with "ตามนโยบายจำลอง", plus the cited document |
| Partly covered | The covered part, then "ไม่มีข้อมูลส่วน… ในคู่มือ" for the rest |
| Not in the handbook | "ขออภัย ไม่มีข้อมูลส่วนนี้ในคู่มือพนักงาน", the list of available topics, and a pointer to HR or Finance |
| Unreadable question | A request to rephrase with two example questions, plus the topic list |
| Ambiguous question | One clarifying question, e.g. "หมายถึงลากิจหรือลาพักร้อน" |
| Injection / out of scope | Polite refusal; no documents or model call involved |

## Stack

| Layer | Choice |
|---|---|
| Chat page | Single HTML file served by FastAPI (`app/web/chat.html`) |
| Admin console | Streamlit (`ui/streamlit_app.py`) |
| API | FastAPI + Pydantic v2 |
| Orchestration | LangGraph (`app/graph/`) |
| LLM | Claude `claude-opus-5-5` via the Anthropic SDK, effort `low`, server-side refusal fallback enabled |
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` (768 dims) run locally with fastembed (ONNX, CPU). Anthropic has no embeddings API. The model is baked into the image, so no network is needed at runtime |
| Vector DB | PostgreSQL 16 + pgvector, exact cosine search |
| Persistence | SQLAlchemy 2, psycopg 3, Alembic |
| Tooling | pip + `requirements.txt`, ruff, mypy (strict), pytest, Docker Compose |

OpenAI / Azure OpenAI are still supported (`LLM_PROVIDER=openai|azure`), and `LLM_PROVIDER=fake` runs fully offline for plumbing checks.

## Quick start

```bash
cp .env.example .env    # then set ANTHROPIC_API_KEY
make up        # build images (first build downloads a ~1 GB embedding model), start db/api/ui, migrate
make ingest    # load data/documents.jsonl
```

Then open http://localhost:8000/.

- If port 5432 is taken, add `DB_HOST_PORT=55499` (any free port) to `.env`. The Makefile reads it from there.
- The API image is about 3.4 GB. Make sure Docker has disk space before `make up`.
- Changing the embedding model or `EMBEDDING_DIMENSIONS` needs `docker compose down -v` (wipes the DB) and a fresh ingest.

## Updating the handbook

1. Edit or replace `data/documents.jsonl`. `data/` is mounted into the containers, so no rebuild is needed.
2. Run `make reset-docs`. It validates the file first and stops before deleting anything if a line is invalid, then removes the old documents and ingests the new ones. Audit logs and evaluation runs are kept.

One JSON object per line. Two formats are accepted:

```json
{"document_id": "HR-002", "title": "นโยบายการลากิจและกรณีเร่งด่วน", "department": "Human Resources", "classification": "INTERNAL", "effective_date": "2026-11-22", "content": "…", "source_type": "policy", "version": "1.0"}
```

```json
{"document_id": "HR-002", "title": "…", "type": "SOP", "category": "leave", "content": "…", "version": 1, "status": "active", "access_level": "internal", "source_authority": "authoritative", "synthetic": true, "updated_at": "2026-11-22"}
```

- `document_id` must look like `HR-002` or `FIN-EXP-003`.
- `classification` / `access_level`: `public` and `internal` are visible to employees, `confidential` to admins only.
- `version` must be a whole number (`"1.0"` is fine, `"1.1"` is rejected).
- Records without a `synthetic` field are treated as synthetic demo data. Real bank documents need a data-handling review first, because their text is sent to Claude.
- To change a document without a full reset, raise its `version` and run `make ingest`. The same version with different content is rejected so edits are never silent.
- PDF and Word files are not read directly yet; convert them to this format first.

Each document becomes one chunk per `##` heading (long sections are split at about 800 characters). The document title is prepended to every chunk.

## Evaluation

`data/evaluation.jsonl` holds 40 cases: 10 direct SOP questions, 10 slang/typo, 5 ambiguous, 5 out of scope, 5 injection, 5 missing knowledge. Each case states the expected behaviour (`answered`, `fallback`, `refused`, `clarify`), the expected document and the facts the answer must contain.

```bash
make eval                     # thresholds 0.25 / 0.28 / 0.35, writes reports/eval_*.json
```

Or use the Evaluation tab in the admin console (log in as `ผู้ดูแลระบบ (demo)`), which also shows failures and exports CSV. Every run calls Claude for each case that passes the input guardrails.

| Metric | Meaning | Good value |
|---|---|---|
| behavior_accuracy | Right action: answer, refuse, fall back or ask back | 1.0 |
| recall_at_5 | Expected document is in the top 5 hits | 1.0 |
| fact_match_rate | Answer contains the expected facts (substring check) | 1.0 |
| false_answer_rate | Answered when it should not have (worst failure) | 0 |
| unnecessary_fallback_rate | Said "no data" when the answer existed | 0 |
| injection_refusal_recall | Injection attempts refused | 1.0 |

The recommended threshold minimises `2 × false_answer_rate + unnecessary_fallback_rate`.

## Admin console (http://localhost:8501/)

Pick `ผู้ดูแลระบบ (demo)` in the sidebar to see every tab.

- **Chat**: ask a question and open *Debug inspector* to see the node sequence, the top-5 hits with similarity, whether the gate passed, the rewritten query, latency and model.
- **Documents**: what is currently ingested, filtered by the caller's access level.
- **Evaluation**: run the 40 cases at several thresholds, inspect failures, download CSV.
- **Audit**: every request with status, reason code, latency and model. Actor and query are stored as HMAC hashes; raw text only when `LOG_RAW_QUERIES=true`.

## API

All endpoints except health and the chat page need `Authorization: Bearer <token>`. Demo tokens: `employee-demo-token`, `admin-demo-token` (override with `DEMO_USERS`). Role and scopes come from the server; extra fields in the request body return 422.

| Method | Path | Access |
|---|---|---|
| GET | `/` | Chat page |
| POST | `/api/v1/chat` | user |
| GET | `/api/v1/documents` | user (filtered by scope) |
| POST | `/api/v1/admin/ingest` | admin |
| POST | `/api/v1/admin/evaluations/run` | admin |
| GET | `/api/v1/admin/evaluations` | admin |
| GET | `/api/v1/admin/audit-logs` | admin |
| GET | `/health/live`, `/health/ready` | open |

```bash
curl -s localhost:8000/api/v1/chat \
  -H 'Authorization: Bearer employee-demo-token' \
  -H 'Content-Type: application/json' \
  -d '{"query": "ลาป๊วยทำไง"}'
```

```json
{
  "status": "answered",
  "answer": "ตามนโยบายจำลอง การลาป่วยให้ยื่นผ่าน HR Portal …",
  "sources": [{"document_id": "HR-001", "title": "นโยบายและขั้นตอนการลาป่วย", "version": 1, "snippet": "…"}],
  "retrieval": {"best_similarity": 0.52, "threshold": 0.28},
  "reason_code": null,
  "decision_type": "document_grounded",
  "debug": {"trace": ["validate_input", "retrieve", "evidence_gate", "rewrite_query", "retrieve", "evidence_gate", "generate", "verify_output", "audit_and_finalize"], "rewritten_query": "ขั้นตอนการลาป่วย", "latency_ms": 4209}
}
```

`status` is `answered | fallback | refused | error`. Every non-answer has `sources: []` and a `reason_code` (`LOW_EVIDENCE`, `MODEL_CANNOT_ANSWER`, `UNCLEAR_QUERY`, `NEEDS_CLARIFICATION`, `PROMPT_INJECTION`, `OUT_OF_SCOPE`, `UNSUPPORTED_CLAIM`, …).

## Project layout

```
app/
  main.py          FastAPI app, request IDs, error schema, health checks, chat page route
  config.py        settings (env vars)
  domain.py        reason codes and the Thai user-facing messages
  graph/           LangGraph state, nodes, edges
  rag/             ingest (+ format normalisation), splitter, embeddings, retriever,
                   rewriter, prompts, generator
  guardrails/      input policy, output verification, PII redaction
  db/              models, repositories
  services/        chat, audit, evaluation
  web/chat.html    employee chat page
data/              documents.jsonl, evaluation.jsonl, source/ (previous versions)
docs/              architecture diagram, demo screenshots, exported evaluation results
reports/           evaluation reports (JSON)
scripts/           ingest.py (--check validates only), run_eval.py
tests/             unit/, integration/, evaluation/
ui/                Streamlit admin console
```

## Development

```bash
make setup               # .venv + requirements-dev.txt + .env
make check               # ruff + mypy strict + 78 unit tests (no network, no DB)
make test-integration    # pgvector tests in a separate rag_test database
make test-live           # real provider calls (costs money)
```

## Security notes

- Retrieved text is wrapped in `<source>` tags, HTML-escaped and treated as data. The rewriter prompt treats the user query the same way. Rewritten text is used only for search; the answer is generated from the original question.
- Access control is a SQL filter on `access_level`, applied before retrieval and again before sources are shown. The topic list in fallback messages uses the same filter. The LLM never decides access.
- Logs store HMAC hashes of actor and query.
- Input rules are defence in depth, not a complete injection defence.
- Containers run as non-root; ports are bound to `127.0.0.1`.

## Known limitations / next steps

- Semantic search only. Add Thai-aware full-text search and a reranker if error analysis shows exact-term misses.
- Typo handling costs one extra Claude call (about 2 s) and only runs when the first search is weak or Claude cannot answer. Questions with no answer in the handbook take 5-7 s because of that retry.
- Each question is answered on its own; there is no conversation memory.
- Numeric grounding catches invented numbers, not invented wording. An independent groundedness check would close that gap.
- The eval set is small and written by the builder. Add a held-out set before reporting accuracy.
- PDF / Word ingestion is not implemented.
- Demo token auth. Production needs SSO/OIDC and real RBAC.
# Mini-Enterprise-RAG
