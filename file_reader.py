import os
import logging
import pdfplumber
import fitz  # PyMuPDF for PDF handling
from docx import Document
from pdf2image import convert_from_path
from docx2pdf import convert
from PIL import Image
from langchain_openai import OpenAI  # OpenAI API for OCR
from langchain.schema import HumanMessage
from dotenv import load_dotenv  # Load environment variables
import base64
from io import BytesIO
import tiktoken  # OpenAI token counter
import time

# Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load OpenAI API Key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the key is loaded correctly
if not openai_api_key:
    raise ValueError("❌ OpenAI API key is missing! Please check your .env file.")

# OpenAI Model Initialization
openai_llm = OpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)

def safe_openai_request(func, *args, max_retries=3):
    """Wraps an OpenAI request to retry on rate limit errors."""
    retries = 0
    while retries < max_retries:
        try:
            return func(*args)
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower():
                wait_time = (2 ** retries) * 5  # Exponential backoff
                print(f"⚠️ Rate limit exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                return f"Error: {str(e)}"
    return "Error: Too many retries. Please try again later."

def encode_image_to_base64(image):
    """Convert a PIL image to a Base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def count_tokens(text):
    """Counts the number of tokens in a given text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def extract_text_from_image(image, max_tokens=5000):
    """Uses OpenAI GPT-4o to extract text from an image, ensuring token limits are not exceeded."""
    base64_image = encode_image_to_base64(image)

    # Create a system prompt
    prompt = "Extract the text from this image accurately."

    # Send image as Base64 to OpenAI
    response = safe_openai_request(
        openai_llm.invoke,
        [HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
        ])]
    )

    extracted_text = response.content.strip() if response else "Error: No text extracted."

    # Check token count and truncate if needed
    if count_tokens(extracted_text) > max_tokens:
        extracted_text = extracted_text[:max_tokens]  # Trim to avoid API rejection
        extracted_text += "\n\n⚠️ Text truncated to avoid exceeding token limits."

    return extracted_text

def clean_text(text):
    """
    Cleans extracted text by removing extra whitespace, line breaks, and metadata.

    Parameters:
        text (str): The extracted text.

    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    text = text.replace("\n", " ").replace("\t", " ")  # Normalize spaces
    text = " ".join(text.split())  # Remove excessive spaces
    return text.strip()

def read_text_file(file_path):
    """Reads and cleans text from a .txt file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return "Error: File not found."

    if not file_path.endswith(".txt"):
        logging.error(f"Unsupported file format: {file_path}")
        return "Error: Unsupported file format."

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logging.warning(f"TXT file is empty: {file_path}")
            return "Error: The uploaded file is empty."

        logging.info(f"Successfully read TXT file: {file_path}")
        return clean_text(content)

    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {str(e)}")
        return f"Error: {str(e)}"

def read_pdf_file(file_path, max_pages):
    """Extracts text from the first `max_pages` of a PDF file."""
    if not os.path.exists(file_path):
        return "Error: File not found."
    if not file_path.endswith(".pdf"):
        return "Error: Unsupported file format."

    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i in range(min(len(pdf.pages), max_pages)):
                extracted_text = pdf.pages[i].extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

        # If no text was extracted, assume the PDF is scanned and use OCR
        if not text.strip():
            logging.warning(f"No text found in {file_path}. Converting to image for OCR.")
            text = ocr_pdf_using_llm(file_path, max_pages)

        return clean_text(text) if text.strip() else "Error: Could not extract text from PDF."
    except Exception as e:
        return f"Error: {str(e)}"

def read_docx_file(file_path, max_pages):
    """Extracts text from the first `max_pages` sections of a DOCX file."""
    if not os.path.exists(file_path):
        return "Error: File not found."
    if not file_path.endswith(".docx"):
        return "Error: Unsupported file format."

    try:
        doc = Document(file_path)
        paragraphs = doc.paragraphs[:max_pages * 50]  # Approximate 50 paragraphs per page
        text = "\n".join([para.text for para in paragraphs if para.text.strip()])

        # If no text is found, assume it's an image-based DOCX and apply OCR
        if not text.strip():
            logging.warning(f"No text found in {file_path}. Converting DOCX to image for OCR.")
            text = ocr_docx_using_llm(file_path)

        return clean_text(text) if text.strip() else "Error: Could not extract text from DOCX."
    except Exception as e:
        return f"Error: {str(e)}"


def ocr_pdf_using_llm(file_path, max_pages=5, batch_size=2):
    """Converts PDF pages to images and extracts text using OpenAI's LLM in smaller batches."""
    try:
        images = convert_from_path(file_path, dpi=300)[:max_pages]  # Convert only `max_pages` pages to images
        extracted_text = ""

        # Process pages in smaller batches to reduce token load
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            for img in batch_images:
                text = extract_text_from_image(img)  # Convert image to Base64 & extract text
                extracted_text += text + "\n"

            # Avoid hitting OpenAI's rate limit (sleep if necessary)
            time.sleep(2)  # Pause for 2 seconds between batches

        return extracted_text if extracted_text.strip() else "Error: OCR failed to extract text."

    except Exception as e:
        return f"Error: {str(e)}"

def ocr_docx_using_llm(file_path):
    """Converts DOCX to PDF, then extracts text using LLM-based OCR."""
    try:
        temp_pdf_path = "temp_docx_conversion.pdf"
        convert(file_path, temp_pdf_path)  # Convert DOCX to PDF
        extracted_text = ocr_pdf_using_llm(temp_pdf_path, max_pages=5)  # Apply OCR on converted PDF
        os.remove(temp_pdf_path)  # Cleanup
        return extracted_text
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text(file_path, max_pages):
    """Determines file type and extracts text accordingly with user-defined `max_pages`."""
    if file_path.endswith(".txt"):
        return read_text_file(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf_file(file_path, max_pages)
    elif file_path.endswith(".docx"):
        return read_docx_file(file_path, max_pages)
    else:
        return "Error: Unsupported file format."
