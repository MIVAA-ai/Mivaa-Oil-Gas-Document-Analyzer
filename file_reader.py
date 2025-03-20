import logging
import os

# Configure logging
logging.basicConfig(
    filename="logs/app.log",  # Save logs to a file
    level=logging.INFO,  # Log INFO and above messages
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def read_text_file(file_path):
    """
    Reads the content of a text file with error handling and logging.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        str: The text content of the file, or an error message if reading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return "Error: File not found."

    if not file_path.endswith(".txt"):  # Ensure only text files are processed
        logging.error(f"Unsupported file format: {file_path}")
        return "Error: Unsupported file format. Please upload a .txt file."

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            logging.warning(f"File is empty: {file_path}")
            return "Error: The uploaded file is empty."

        logging.info(f"File successfully read: {file_path}")
        return content

    except PermissionError:
        logging.error(f"Permission denied: {file_path}")
        return "Error: Permission denied while reading the file."

    except Exception as e:
        logging.error(f"Unexpected error reading file {file_path}: {str(e)}")
        return f"Error: An unexpected issue occurred - {str(e)}"
