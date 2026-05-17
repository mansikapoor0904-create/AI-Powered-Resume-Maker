from flask import Flask, render_template, request
import os
import PyPDF2
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


# Extract text from PDF
def extract_text_from_pdf(pdf_path):

    text = ""

    try:
        with open(pdf_path, 'rb') as file:

            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

    except:
        print("Error reading PDF")

    return text


# Preprocess text
def preprocess_text(text):

    doc = nlp(text.lower())

    tokens = []

    for token in doc:

        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)

    return " ".join(tokens)


# Rank resumes
def rank_resumes(job_description, resumes):

    documents = [job_description] + resumes

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )

    return similarity_scores[0]


@app.route('/', methods=['GET', 'POST'])
def index():

    ranked_data = []

    if request.method == 'POST':

        job_description = request.form['job_description']

        uploaded_files = request.files.getlist('resumes')

        resume_texts = []
        file_names = []

        for file in uploaded_files:

            if file.filename.endswith('.pdf'):

                filepath = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    file.filename
                )

                file.save(filepath)

                extracted_text = extract_text_from_pdf(filepath)

                if extracted_text.strip() != "":

                    processed_text = preprocess_text(extracted_text)

                    resume_texts.append(processed_text)

                    file_names.append(file.filename)

        # SAFETY CHECK
        if len(resume_texts) > 0:

            processed_job_description = preprocess_text(
                job_description
            )

            scores = rank_resumes(
                processed_job_description,
                resume_texts
            )

            for i in range(len(file_names)):

                ranked_data.append({
                    'name': file_names[i],
                    'score': round(scores[i] * 100, 2)
                })

            ranked_data = sorted(
                ranked_data,
                key=lambda x: x['score'],
                reverse=True
            )

    return render_template(
        'index.html',
        ranked_data=ranked_data
    )


if __name__ == '__main__':
    app.run(debug=True, port=8000)