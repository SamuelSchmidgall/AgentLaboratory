import random, time
from functools import lru_cache

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
from pypdf import PdfReader
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///papers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Paper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    text = db.Column(db.Text, nullable=True)

def update_papers_from_uploads():
    for _tries in range(5):
        try:
            uploads_dir = app.config['UPLOAD_FOLDER']
            file_list = os.listdir(uploads_dir)
            print("Files in uploads folder:", file_list)
            for filename in file_list:
                if filename.lower().endswith('.pdf'):
                    # Check if file is already in the DB
                    if not Paper.query.filter_by(filename=filename).first():
                        print("Processing file:", filename)
                        file_path = os.path.join(uploads_dir, filename)
                        extracted_text = ""
                        try:
                            reader = PdfReader(file_path)
                            for page in reader.pages:
                                text = page.extract_text()
                                if text:
                                    extracted_text += text
                        except Exception as e:
                            flash(f'Error processing {filename}: {e}')
                            continue
                        if not extracted_text.strip():
                            print(f"Warning: No text extracted from {filename}")
                        else:
                            print(f"Extracted {len(extracted_text)} characters from {filename}")
                        new_paper = Paper(filename=filename, text=extracted_text)
                        db.session.add(new_paper)
            db.session.commit()
            return
        except Exception as e:
            print("WEB SERVER LOAD EXCEPTION", e, str(e))
            time.sleep(random.randint(5, 15))
    return
    #raise Exception("FAILED TO UPDATE")

@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/update', methods=['GET'])
def update_on_demand():
    update_papers_from_uploads()
    return jsonify({"message": "Uploads folder processed successfully."})

@app.route('/')
def index():
    update_papers_from_uploads()
    papers = Paper.query.all()
    return render_template('index.html', papers=papers)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            extracted_text = ""
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text
            except Exception as e:
                flash(f'Error processing PDF: {e}')
            new_paper = Paper(filename=filename, text=extracted_text)
            db.session.add(new_paper)
            db.session.commit()
            flash('File uploaded and processed successfully!')
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if query:
        model = get_embedding_model()
        papers = Paper.query.all()
        query_embedding = model.encode([query])
        paper_texts = [paper.text for paper in papers if paper.text]
        if not paper_texts:
            return render_template('search.html', papers=[], query=query)
        paper_embeddings = model.encode(paper_texts)
        similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        papers_with_scores = list(zip([p for p in papers if p.text], similarities))
        papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
        return render_template('search.html', papers=papers_sorted, query=query)
    return render_template('search.html', papers=[], query=query)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    model = get_embedding_model()
    papers = Paper.query.all()
    if not papers:
        return jsonify({'query': query, 'results': []})
    query_embedding = model.encode([query])
    paper_texts = [paper.text for paper in papers if paper.text]
    if not paper_texts:
        return jsonify({'query': query, 'results': []})
    paper_embeddings = model.encode(paper_texts)
    similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
    papers_with_scores = list(zip([p for p in papers if p.text], similarities))
    papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
    results = []
    for paper, score in papers_sorted:
        pdf_url = url_for('uploaded_file', filename=paper.filename, _external=True)
        results.append({
            'id': paper.id,
            'filename': paper.filename,
            'similarity': float(score),
            'pdf_url': pdf_url
        })
    return jsonify({'query': query, 'results': results})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

@app.route('/view/<int:paper_id>')
def view_pdf(paper_id):
    paper = Paper.query.get_or_404(paper_id)
    pdf_url = url_for('uploaded_file', filename=paper.filename, _external=True)
    return render_template('view.html', paper=paper, pdf_url=pdf_url)


def run_app(port=5000):
    # Reset the database by removing the existing file
    db_path = "papers.db"
    if os.path.exists("instance/" + db_path):
        os.remove("instance/" + db_path)
    with app.app_context():
        db.create_all()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=False, port=port)

if __name__ == '__main__':
    run_app()
