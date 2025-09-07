# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
import os
import io
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
from collections import Counter
import logging

# Download required NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB max file size

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class DocumentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def extract_text_from_image(self, file_path):
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            raise Exception(f"Failed to extract text from image: {str(e)}")

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", "", text)
        return text.strip()

    def split_into_sentences(self, text):
        """Split text into sentences"""
        sentences = sent_tokenize(text)
        # Filter out very short sentences
        return [
            s.strip() for s in sentences if len(s.strip()) > 10 and len(s.split()) > 3
        ]

    def calculate_word_frequencies(self, text):
        """Calculate word frequencies for scoring"""
        words = word_tokenize(text.lower())
        words = [
            word for word in words if word.isalnum() and word not in self.stop_words
        ]
        return FreqDist(words)

    def score_sentence(self, sentence, sentences, word_freq, text_length):
        """Score a sentence based on various factors"""
        words = [word for word in word_tokenize(sentence.lower()) if word.isalnum()]
        score = sum(word_freq[word] for word in words if word in word_freq and word not in self.stop_words)
        word_count = len(words)
        if 8 <= word_count <= 25:
            score += 2
        elif 5 <= word_count <= 30:
            score += 1
        sentence_index = sentences.index(sentence)
        total_sentences = len(sentences)
        if sentence_index in [0, total_sentences - 1]:
            score += 2
        elif sentence_index < total_sentences * 0.1:
            score += 1
        important_keywords = [
            "important", "significant", "key", "main", "primary", "crucial", "essential", "major", "fundamental", "critical", "summary", "conclusion", "result", "finding", "therefore", "however"
        ]
        sentence_lower = sentence.lower()
        score += sum(1.5 for keyword in important_keywords if keyword in sentence_lower)

        # Numbers and dates scoring
        if re.search(r"\d+", sentence):
            score += 0.5

        # Proper nouns scoring
        proper_nouns = len(re.findall(r"\b[A-Z][a-z]+\b", sentence))
        score += min(proper_nouns * 0.3, 1.5)

        # Sentence connectivity (sentences with conjunctions)
        connectives = [
            "therefore",
            "however",
            "moreover",
            "furthermore",
            "additionally",
        ]
        for connective in connectives:
            if connective in sentence_lower:
                score += 0.5

        return score

    def extract_key_points(self, text, sentences, max_points=5):
        """Extract key points from the text"""
        key_points = []

        # Look for numbered lists
        numbered_pattern = r"(?:^|\n)\s*\d+[\.\)]\s*([^\n]+)"
        numbered_items = re.findall(numbered_pattern, text, re.MULTILINE)
        if numbered_items:
            key_points.extend([item.strip() for item in numbered_items[:max_points]])

        # Look for bullet points
        if len(key_points) < max_points:
            if bullet_items := re.findall(r"(?:^|\n)\s*[•\-\*]\s*([^\n]+)", text, re.MULTILINE):
                key_points.extend([item.strip() for item in bullet_items[:max_points - len(key_points)]])

        # If no structured lists, find sentences with high keyword density
        if len(key_points) < max_points:
            keywords = ["key", "important", "main", "primary", "significant", "crucial", "essential"]
            keyword_sentences = [
                (sentence, sum(keyword in sentence.lower() for keyword in keywords))
                for sentence in sentences if len(sentence) < 150
            ]
            keyword_sentences = [ks for ks in keyword_sentences if ks[1] > 0]
            keyword_sentences.sort(key=lambda x: x[1], reverse=True)
            key_points.extend([sent[0] for sent in keyword_sentences[:max_points - len(key_points)]])

        # Fallback: use first few sentences if still empty
        if not key_points:
            key_points = [s for s in sentences[:3] if len(s) < 150]

        return key_points[:max_points]

    def generate_summary(self, text, summary_length="medium"):
        """Generate summary of the given text"""
        if not text or len(text.strip()) < 50:
            raise Exception("Text is too short or empty to summarize")

        # Clean text
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)

        if not sentences:
            raise Exception("No valid sentences found in the text")

        # Calculate word frequencies
        word_freq = self.calculate_word_frequencies(cleaned_text)

        # Determine target number of sentences
        total_sentences = len(sentences)
        if summary_length == "short":
            target_sentences = max(1, min(3, int(total_sentences * 0.1)))
        elif summary_length == "medium":
            target_sentences = max(2, min(6, int(total_sentences * 0.2)))
        else:  # long
            target_sentences = max(3, min(10, int(total_sentences * 0.3)))

        # Score sentences
        scored_sentences = []
        for sentence in sentences:
            score = self.score_sentence(
                sentence, sentences, word_freq, len(cleaned_text)
            )
            scored_sentences.append((sentence, score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:target_sentences]
        summary_sentences = [(sentence, sentences.index(sentence)) for sentence, _ in top_sentences]
        summary_sentences.sort(key=lambda x: x[1])

        # Extract key points
        key_points = self.extract_key_points(cleaned_text, sentences)

        return {
            "summary": " ".join([sent[0] for sent in summary_sentences]),
            "key_points": key_points,
            "original_sentences": len(sentences),
            "summary_sentences": len(summary_sentences),
            "word_count": len(word_tokenize(cleaned_text)),
        }


# Initialize document processor
doc_processor = DocumentProcessor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": "File type not supported. Only PDF and image files are allowed."
                    }
                ),
                400,
            )

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Extract text based on file type
        file_extension = filename.rsplit(".", 1)[1].lower()

        if file_extension == "pdf":
            extracted_text = doc_processor.extract_text_from_pdf(file_path)
        else:
            extracted_text = doc_processor.extract_text_from_image(file_path)

        # Clean up uploaded file
        os.remove(file_path)

        if not extracted_text or len(extracted_text.strip()) < 50:
            return (
                jsonify(
                    {
                        "error": "Could not extract sufficient text from the document. The document may be empty or the text may be unclear."
                    }
                ),
                400,
            )

        return jsonify(
            {
                "success": True,
                "text": extracted_text,
                "filename": filename,
                "file_size": len(file.read()),
                "file_type": file.content_type,
            }
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500


@app.route("/summarize", methods=["POST"])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "")
        summary_length = data.get("length", "medium")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        summary_result = doc_processor.generate_summary(text, summary_length)

        return jsonify(
            {
                "success": True,
                "summary": summary_result["summary"],
                "key_points": summary_result["key_points"],
                "stats": {
                    "original_sentences": summary_result["original_sentences"],
                    "summary_sentences": summary_result["summary_sentences"],
                    "word_count": summary_result["word_count"],
                    "compression_ratio": round(
                        (
                            summary_result["summary_sentences"]
                            / summary_result["original_sentences"]
                        )
                        * 100,
                        1,
                    ),
                },
            }
        )

    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
