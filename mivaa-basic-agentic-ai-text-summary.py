import os
from dotenv import load_dotenv
from typing import TypedDict, List  # Importing type hints for better code clarity
from langgraph.graph import StateGraph, END  # Importing StateGraph for managing states
from langchain.prompts import PromptTemplate  # Importing PromptTemplate to format prompts
from langchain_openai import ChatOpenAI  # Importing ChatOpenAI to use OpenAI's chat model
from langchain.schema import HumanMessage  # Importing HumanMessage to represent user input
import logging
import gradio as gr
import logging
from file_reader import read_text_file

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


def summarize_text(state):
    """
    Generates a one-sentence summary of the input oil and gas document.

    Parameters:
        state (dict): The current state dictionary containing the input text.

    Returns:
        dict: A dictionary with the "summary" key containing the summarized text.
    """
    if not state.get("text") or not state["text"].strip():
        logging.warning("Empty input text received for summarization.")
        return {"summary": "Error: No text provided for summarization."}

    try:
        # Define an improved summarization prompt
        summarization_prompt = PromptTemplate.from_template(
            """Summarize the following oil and gas industry document in one concise sentence.

            Text: {text}

            Summary:"""
        )

        # Create a processing chain: Pass the formatted prompt to the LLM
        chain = summarization_prompt | llm

        # Execute the chain using the text from the state dictionary
        response = chain.invoke({"text": state["text"]})

        # Ensure response is valid
        if not response or not response.content:
            logging.error("LLM returned an empty summary response.")
            return {"summary": "Error: Summary could not be generated."}

        summary = response.content.strip()
        logging.info(f"Generated Summary: {summary}")

        return {"summary": summary}

    except Exception as e:
        logging.error(f"Error in summarize_text: {str(e)}")
        return {"summary": f"Error: {str(e)}"}


def process_uploaded_file(file):
    """
    Reads the uploaded text file and processes it through the LangChain workflow.

    Parameters:
        file (gr.File): The uploaded text file.

    Returns:
        tuple: Classification, Entities, and Summary results.
    """
    if file is None:
        logging.warning("No file uploaded.")
        return "Error: No file uploaded.", "Error: No file uploaded.", "Error: No file uploaded."

    # Check file size
    file_size_mb = os.path.getsize(file.name) / (1024 * 1024)  # Convert bytes to MB
    if file_size_mb > MAX_FILE_SIZE_MB:
        logging.error(f"File {file.name} exceeds the 5MB limit ({file_size_mb:.2f}MB).")
        return "Error: File too large. Please upload a file smaller than 5MB.", "", ""

    logging.info(f"Processing file: {file.name} ({file_size_mb:.2f}MB)")

    # Read file content
    text_content = read_text_file(file.name)

    if text_content.startswith("Error:"):
        logging.error(f"File reading error: {text_content}")
        return text_content, "Error extracting entities.", "Error generating summary."

    logging.info(f"Processing file: {file.name}")

    # Define a state-based workflow using a StateGraph
    workflow = StateGraph(State)

    # Add processing nodes (functions) to the workflow graph
    workflow.add_node("classification_node", classification_node)  # Step 1: Classify text
    workflow.add_node("entity_extraction", entity_extraction_node)  # Step 2: Extract entities
    workflow.add_node("summarization", summarize_text)  # Step 3: Summarize the text

    # Define the execution sequence by setting edges between nodes
    workflow.set_entry_point("classification_node")  # Start with classification
    workflow.add_edge("classification_node", "entity_extraction")  # Then extract entities
    workflow.add_edge("entity_extraction", "summarization")  # Then summarize
    workflow.add_edge("summarization", END)  # Mark the end of the workflow

    # Compile the workflow into an executable application
    app = workflow.compile()

    # Initialize the state dictionary with the file content
    state_input = {"text": text_content}

    # Execute the LangChain workflow
    try:
        result = app.invoke(state_input)

        classification = result.get("classification", "⚠️ Error: Classification failed")
        entities = result.get("entities", "⚠️ Error: Entity extraction failed")
        summary = result.get("summary", "⚠️ Error: Summarization failed")

        logging.info(f"Results - Classification: {classification}, Entities: {entities}, Summary: {summary}")

        return classification, entities, summary, "✅ Processing complete!"

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return "⚠️ Error: Processing failed.", "", "", f"❌ {str(e)}"


# Create the Gradio UI
with gr.Blocks(theme="default") as demo:
    gr.Markdown(
        """
        # 📄 Oil & Gas Document Analyzer  
        Upload a **.txt file** (Max: **5MB**) and get:
        - 📌 **Document Classification**  
        - 🔍 **Key Entity Extraction**  
        - ✍️ **Summarization**  

        ⚡ Powered by **LangChain + OpenAI GPT**
        """,
        elem_id="header"
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 Upload your .txt file", file_types=[".txt"])

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="📢 Status", interactive=False, lines=1)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📌 Document Classification")
                output_classification = gr.Textbox(label="", interactive=False)

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🔍 Extracted Entities")
                output_entities = gr.Textbox(label="", interactive=False, lines=5)

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ✍️ Summary")
                output_summary = gr.Textbox(label="", interactive=False, lines=3)

    with gr.Row():
        process_button = gr.Button("🚀 Process File")

    # Run the processing function on button click
    process_button.click(
        process_uploaded_file,
        inputs=file_input,
        outputs=[output_classification, output_entities, output_summary, status_output]
    )

    gr.Markdown(
        "🚀 **Developed by Mivaa** | [🔗 Visit Website](https://deepdatawithmivaa.com/) | 🌍 **AI-Powered Solutions**",
        elem_id="footer"
    )

# Launch the Gradio app
if __name__ == "__main__":
    logging.info("Starting Gradio UI...")
    demo.launch()