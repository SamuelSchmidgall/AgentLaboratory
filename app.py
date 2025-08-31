# app.py
import os
import time
import random
import numpy as np

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, jsonify, has_request_context
)
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

from pypdf import PdfReader  # unified parser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# App configuration
# ----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-not-secret")
app.config["UPLOAD_FOLDER"] = "uploads/"
# Store DB inside instance/ to keep it out of source and align with resets
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///instance/papers.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {"pdf"}

# Preload a compact sentence-transformer
model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------
# Database model
# ----------------------------
class Paper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), unique=True, nullable=False)
    text = db.Column(db.Text, nullable=True)
    # New: store embeddings once at ingest for fast search
    embedding = db.Column(db.LargeBinary, nullable=True)


# ----------------------------
# Utilities
# ----------------------------
def safe_flash(msg, category="message"):
    if has_request_context():
        flash(msg, category)
    else:
        app.logger.warning(f"[flash skipped] {msg}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_dirs():
    os.makedirs("instance", exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def ensure_embedding_column():
    """Add 'embedding' column if missing (SQLite allows simple ALTER ADD)."""
    with app.app_context():
        inspector = db.inspect(db.engine)
        cols = [c["name"] for c in inspector.get_columns("paper")]
        if "embedding" not in cols:
            with db.engine.connect() as conn:
                conn.execute(db.text("ALTER TABLE paper ADD COLUMN embedding BLOB"))
            app.logger.info("Added 'embedding' column to 'paper' table.")


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                text += page_text
            except Exception as e:
                app.logger.warning(f"Page extract failed for {file_path}: {e}")
    except Exception as e:
        safe_flash(f"Error reading PDF {os.path.basename(file_path)}: {e}")
    return text


def compute_embedding(text: str) -> np.ndarray:
    # SentenceTransformer returns numpy array already; standardize dtype
    return model.encode([text])[0].astype(np.float32)


def _add_or_update_paper(filename: str, text: str):
    paper = Paper.query.filter_by(filename=filename).first()
    emb = None
    if text and text.strip():
        emb = compute_embedding(text)
    if paper is None:
        paper = Paper(filename=filename, text=text, embedding=(emb.tobytes() if emb is not None else None))
        db.session.add(paper)
    else:
        paper.text = text
        if paper.embedding is None and emb is not None:
            paper.embedding = emb.tobytes()
    db.session.commit()


def update_papers_from_uploads():
    """One-pass ingest of PDFs in uploads/ with retries; safe to call ad-hoc."""
    uploads_dir = app.config["UPLOAD_FOLDER"]
    for _tries in range(5):
        try:
            file_list = os.listdir(uploads_dir)
            app.logger.info(f"Files in uploads folder: {file_list}")
            for filename in file_list:
                if not filename.lower().endswith(".pdf"):
                    continue
                # Skip if already in DB with embedding
                paper = Paper.query.filter_by(filename=filename).first()
                if paper and paper.embedding is not None:
                    continue
                file_path = os.path.join(uploads_dir, filename)
                try:
                    extracted_text = extract_text_from_pdf(file_path)
                    if not extracted_text.strip():
                        app.logger.warning(f"No text extracted from {filename}")
                    _add_or_update_paper(filename, extracted_text)
                    app.logger.info(f"Ingested {filename}")
                except Exception as e:
                    safe_flash(f"Error processing {filename}: {e}")
                    continue
            return
        except Exception as e:
            app.logger.error(f"WEB SERVER LOAD EXCEPTION: {e}")
            time.sleep(random.randint(5, 15))
    return


def get_embeddings_matrix_and_papers():
    """Return NxD float32 matrix and aligned Paper list; backfill missing embeds."""
    papers = Paper.query.all()
    vectors = []
    changed = False
    for p in papers:
        if p.text and p.embedding is None:
            # backfill
            try:
                p.embedding = compute_embedding(p.text).tobytes()
                changed = True
            except Exception as e:
                app.logger.warning(f"Embedding backfill failed for {p.filename}: {e}")
                p.embedding = None
        vec = None
        if p.embedding:
            vec = np.frombuffer(p.embedding, dtype=np.float32)
        else:
            # empty vector fallback (rare)
            vec = np.zeros((384,), dtype=np.float32)
        vectors.append(vec)
    if changed:
        db.session.commit()
    if len(vectors) == 0:
        return np.zeros((0, 384), dtype=np.float32), []
    mat = np.vstack(vectors)
    return mat, papers


# ----------------------------
# Routes
# ----------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/update", methods=["GET"])
def update_on_demand():
    update_papers_from_uploads()
    return jsonify({"message": "Uploads folder processed successfully."})


@app.route("/")
def index():
    update_papers_from_uploads()
    papers = Paper.query.all()
    return render_template("index.html", papers=papers)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "pdf" not in request.files:
            safe_flash("No file part")
            return redirect(request.url)
        file = request.files["pdf"]
        if file.filename == "":
            safe_flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            try:
                extracted_text = extract_text_from_pdf(file_path)
                _add_or_update_paper(filename, extracted_text)
                safe_flash("File uploaded and processed successfully!")
            except Exception as e:
                safe_flash(f"Error processing PDF: {e}")
            return redirect(url_for("index"))
        else:
            safe_flash("Unsupported file type. Please upload a PDF.")
            return redirect(request.url)
    return render_template("upload.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("k", "10"))
    results = []
    if query:
        emb_mat, papers = get_embeddings_matrix_and_papers()
        if len(papers) > 0:
            qv = model.encode([query]).astype(np.float32)
            sims = cosine_similarity(qv, emb_mat)[0]
            order = np.argsort(-sims)
            for idx in order[:top_k]:
                p = papers[idx]
                results.append((p, float(sims[idx])))
    return render_template("search.html", query=query, results=results)


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    top_k = int(request.args.get("k", "10"))
    emb_mat, papers = get_embeddings_matrix_and_papers()
    if len(papers) == 0:
        return jsonify({"query": query, "results": []})
    qv = model.encode([query]).astype(np.float32)
    sims = cosine_similarity(qv, emb_mat)[0]
    order = np.argsort(-sims)
    results = []
    for idx in order[:top_k]:
        p = papers[idx]
        pdf_url = url_for("uploaded_file", filename=p.filename, _external=True)
        results.append({
            "id": p.id,
            "filename": p.filename,
            "similarity": float(sims[idx]),
            "pdf_url": pdf_url
        })
    return jsonify({"query": query, "results": results})


@app.route("/api/papers")
def api_papers():
    papers = Paper.query.all()
    return jsonify([
        {"id": p.id, "filename": p.filename, "has_text": bool(p.text), "has_embedding": p.embedding is not None}
        for p in papers
    ])


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # Only serve PDFs from the uploads directory
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, mimetype="application/pdf")


@app.route("/view/<int:paper_id>")
def view_pdf(paper_id):
    paper = Paper.query.get_or_404(paper_id)
    pdf_url = url_for("uploaded_file", filename=paper.filename, _external=True)
    return render_template("view.html", paper=paper, pdf_url=pdf_url)


# ----------------------------
# Entrypoint
# ----------------------------
def run_app(port=5000):
    ensure_dirs()
    # Optional reset controlled by env var
    if os.getenv("RESET_DB_ON_START", "false").lower() == "true":
        db_path = os.path.join("instance", "papers.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            app.logger.warning("Resetting database on start.")

    with app.app_context():
        db.create_all()
        ensure_embedding_column()
    app.run(debug=False, port=port)


if __name__ == "__main__":
    run_app()
