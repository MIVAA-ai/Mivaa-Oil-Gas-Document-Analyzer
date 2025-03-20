import os
from dotenv import load_dotenv
from typing import TypedDict, List  # Importing type hints for better code clarity
from langgraph.graph import StateGraph, END  # Importing StateGraph for managing states
from langchain.prompts import PromptTemplate  # Importing PromptTemplate to format prompts
from langchain_openai import ChatOpenAI  # Importing ChatOpenAI to use OpenAI's chat model
from langchain.schema import HumanMessage  # Importing HumanMessage to represent user input
import gradio as gr
import logging
from file_reader import extract_text

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Maximum allowed file size (5MB)
MAX_FILE_SIZE_MB = 5  # 5MB in megabytes

# StateGraph manage information flow between agent components.
# PromptTemplate creates consistent instructions, and ChatOpenAI connects to OpenAI’s chat models to power agent’s thinking.

# Load environment variables
load_dotenv()

# Ensure API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("⚠️ OPENAI_API_KEY is missing! Add it to the .env file.")

# Define a TypedDict to represent the state of the agent
class State(TypedDict):
    text: str  # Stores the original question or task
    classification: str  # Tracks the agent's thinking and decisions
    entities: List[str]  # Stores intermediate results from tools (like extracted entities)
    summary: str  # Holds the final summary or response

# Initialize the language model (LLM) from OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Specifies the OpenAI model to use (GPT-4o Mini for cost-efficiency)
    temperature=0,  # Sets the response randomness; 0 ensures deterministic and consistent answers
    openai_api_key=openai_api_key  # Pass API key securely
)


def classification_node(state: State):
    """
    Dynamically classifies an oil and gas industry document based on its content.

    The model will determine the most appropriate category instead of choosing from a fixed list.

    Returns:
        dict: A dictionary with the "classification" key containing the detected category.
    """

    if not state.get("text") or not state["text"].strip():
        logging.warning("Empty input text received for classification.")
        return {"classification": "Error: No text provided for classification."}

    try:
        # Define a dynamic classification prompt
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following oil and gas industry document and determine the most appropriate category for it.

            Do NOT assume predefined categories. Instead, dynamically assign a category that best describes the document 
            based on its content.

            Provide the response as:
            Category: <Detected Category>

            Example:
            Category: Drilling Operations Report
            Category: Reservoir Analysis
            Category: Seismic Data Interpretation
            Category: Regulatory Filing

            Ensure that the category is relevant and concise.

            Text: {text}

            Category:
            """
        )

        # Format the prompt with the input text from the state
        message = HumanMessage(content=prompt.format(text=state["text"]))

        # Invoke the OpenAI language model
        response = llm.invoke([message]).content.strip()

        if not response:
            logging.error("LLM returned an empty classification response.")
            return {"classification": "Error: Could not determine document category."}

        # Ensure response is structured properly
        category = response.replace("Category:", "").strip()  # Clean up response format

        logging.info(f"Detected Category: {category}")

        return {"classification": category}

    except Exception as e:
        logging.error(f"Error in classification_node: {str(e)}")
        return {"classification": f"Error: {str(e)}"}


def entity_extraction_node(state: State):
    """
    Dynamically extracts named entities from the input text without predefined categories.

    The LLM will determine relevant entity types based on the document context.

    Returns:
        dict: A dictionary with the "entities" key containing a structured labeled text string.
    """

    if not state.get("text") or not state["text"].strip():
        logging.warning("Empty input text received for entity extraction.")
        return {"entities": "Error: No text provided for entity extraction."}

    try:
        # Define a dynamic entity extraction prompt
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following oil and gas industry document and dynamically identify key entities.

            Do NOT assume a fixed set of entity categories. Instead, extract only the relevant entities 
            present in the document and label them accordingly.

            Format the output as follows:

            <Entity Type>: <comma-separated list of extracted entities>

            Example:
            Company: Shell, ExxonMobil, BP
            Well Name: Well A-12, Exploration Well B-27
            Regulatory Agency: EPA, OGP, DOE
            Equipment: Rig Stena Carron, Platform Brent Delta

            If no entities of a particular type exist, simply exclude that category from the response.

            Text: {text}

            Extracted Entities:
            """
        )

        # Format the prompt with the input text from the state
        message = HumanMessage(content=prompt.format(text=state["text"]))

        # Invoke the OpenAI language model
        response = llm.invoke([message]).content.strip()

        if not response:
            logging.error("LLM returned an empty entity extraction response.")
            return {"entities": "Error: No entities extracted."}

        # Ensure response is structured properly
        formatted_entities = response.strip()

        logging.info(f"Extracted Entities:\n{formatted_entities}")

        return {"entities": formatted_entities}

    except Exception as e:
        logging.error(f"Error in entity_extraction_node: {str(e)}")
        return {"entities": f"Error: {str(e)}"}

import time

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "openai_usage.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def summarize_text(state):
    """Generates a summary based on the user's selected summary length and logs response time & token usage."""

    summary_length_map = {
        "Short": "one sentence",
        "Detailed": "three paragraphs"
    }

    # Get user's summary preference
    summary_length = state.get("summary_length", "Short")  # Default to Short
    summary_length = summary_length_map.get(summary_length, "one sentence")  # Ensure valid mapping

    # Create a prompt dynamically based on user preference
    prompt = f"Summarize the following text in {summary_length}.\n\nText: {state['text']}\n\nSummary:"

    try:
        start_time = time.time()  # Start tracking time

        response = llm.invoke([HumanMessage(content=prompt)])  # Call OpenAI LLM

        end_time = time.time()  # End tracking time
        elapsed_time = round(end_time - start_time, 2)  # Calculate elapsed time

        # ✅ Extract text from AIMessage
        if isinstance(response, str):
            summary = response.strip()
        elif hasattr(response, "content"):  # AIMessage case
            summary = response.content.strip()
        else:
            summary = "⚠️ Error: Unexpected LLM response format."

        # ✅ Extract token usage details if available
        token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)

        # ✅ Log API usage and response time
        logging.info(
            f"🔹 Summary Generated | Time: {elapsed_time}s | Tokens Used: Input={input_tokens}, Output={output_tokens}, Total={total_tokens}"
        )

    except Exception as e:
        summary = f"⚠️ Error: Failed to generate summary - {str(e)}"
        logging.error(f"❌ OpenAI Request Failed: {str(e)}")

    return {"summary": summary}

# Global variable to store the compiled workflow
app = None

def initialize_workflow():
    """Initializes the LangGraph workflow for text processing."""
    global app
    if app is None:  # Ensure the workflow is only initialized once
        workflow = StateGraph(State)

        # Add processing nodes (functions) to the workflow graph
        workflow.add_node("classification_node", classification_node)
        workflow.add_node("entity_extraction", entity_extraction_node)
        workflow.add_node("summarization", summarize_text)

        # Define execution order
        workflow.set_entry_point("classification_node")
        workflow.add_edge("classification_node", "entity_extraction")
        workflow.add_edge("entity_extraction", "summarization")
        workflow.add_edge("summarization", END)

        # Compile the workflow once
        app = workflow.compile()

# Initialize workflow at script startup
initialize_workflow()


def process_uploaded_file(file, max_pages, max_file_size, summary_length):
    """
    Reads the uploaded file and processes it through LLM with OCR.
    Returns classification, entity extraction, summary, OCR status, and status message.
    """

    if file is None:
        logging.warning("No file uploaded.")
        return "⚠️ Error: No file uploaded.", "", "", "❌ No OCR Applied.", "❌ No file uploaded."

    # Ensure LangGraph workflow is initialized
    global app
    if app is None:
        initialize_workflow()  # Call the function to initialize if not already

    logging.info(f"Processing file: {file.name}")

    # Read file content
    text_content = extract_text(file.name, max_pages)
    if text_content.startswith("Error:"):
        return text_content, "", "", "❌ No OCR Applied.", "❌ Error in file processing."

    # Detect OCR usage
    is_ocr_applied = "✅ OCR Applied for Scanned Document" if "OCR" in text_content else "No OCR Applied"

    # Execute LangChain workflow
    try:
        state_input = {"text": text_content, "summary_length": summary_length}

        result = app.invoke(state_input)

        classification = result.get("classification", "⚠️ Error: Classification failed")
        entities = result.get("entities", "⚠️ Error: Entity extraction failed")
        summary = result.get("summary", "⚠️ Error: Summarization failed")

        logging.info(f"Results - Classification: {classification}, Entities: {entities}, Summary: {summary}")

        return classification, entities, summary, is_ocr_applied, "✅ Processing complete!"

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return "⚠️ Error: Processing failed.", "", "", "❌ No OCR Applied.", f"❌ {str(e)}"


# Create the Gradio UI
with gr.Blocks(theme="default") as demo:
    gr.Markdown(
        """
        # 📄 AI-Powered Document Analyzer  
        Upload a **.txt, .pdf, or .docx** file and get:
        - 📌 **Document Classification**  
        - 🔍 **Key Entity Extraction**  
        - ✍️ **Summarization**  
        - 🖼️ **LLM-Based OCR for Scanned PDFs & DOCX Files**  

        **ℹ️ Note:** If your document is a **scanned PDF or image-based DOCX**, our system will **automatically apply LLM-based OCR** to extract text.  
        ⚡ Powered by **LangChain + LangGraph + OpenAI**
        """
    )

    summary_length_choice = gr.Radio(
        choices=["Short", "Detailed"],
        label="✍️ Summary Length",
        value="Short"
    )

    with gr.Row():
        file_input = gr.File(label="📂 Upload your .txt, .pdf, or .docx file", file_types=[".txt", ".pdf", ".docx"])
        max_pages_input = gr.Slider(5, 50, value=15, step=5, label="📄 Max Pages to Read")
        max_file_size_input = gr.Slider(1, 50, value=5, step=1, label="📁 Max File Size (MB)")
        status_output = gr.Textbox(label="📢 Status", interactive=False, lines=1)
        ocr_status_output = gr.Textbox(label="🖼️ OCR Status", interactive=False, lines=1)

    with gr.Row():
        output_classification = gr.Textbox(label="📌 Classification", interactive=False)
        output_entities = gr.Textbox(label="🔍 Extracted Entities", interactive=False, lines=5)
        output_summary = gr.Textbox(label="✍️ Summary", interactive=False, lines=3)

    process_button = gr.Button("🚀 Process File")
    process_button.click(
        process_uploaded_file,
        inputs=[file_input, max_pages_input, max_file_size_input, summary_length_choice],
        outputs=[output_classification, output_entities, output_summary, ocr_status_output, status_output]
    )

    gr.Markdown(
        "🚀 **Developed by Mivaa** | [🔗 Visit Website](https://deepdatawithmivaa.com/) | 🌍 **AI-Powered Solutions**",
        elem_id="footer"
    )

# Launch the Gradio app
if __name__ == "__main__":
    logging.info("Starting Gradio UI...")
    demo.launch()