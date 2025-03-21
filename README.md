# AI-Powered Document Processor for Energy Data  
### **Automated Classification, Entity Extraction, and Summarization for Oil & Gas Documents**  

## 📌 Overview  
This AI-powered document processor leverages **LangGraph, OpenAI’s GPT-4o, and Gradio** to automate **document classification, entity extraction, summarization, and OCR processing** for energy sector documents such as:  

- **Well Reports & Logs**  
- **Reservoir & Subsurface Studies**  
- **Operations & Production Data**  
- **Regulatory & Compliance Documents**  
- **Research & Technical Papers**  
- **Business & Market Reports**  
- **Contracts & Legal Agreements**  

This tool is designed to **streamline data workflows** in oil & gas data management by turning raw documents into structured insights.  

---

## 📌 Features  

✔ **AI-Based Document Classification** – Dynamically classifies documents without fixed categories.  
✔ **Entity Extraction** – Identifies well names, reservoirs, regulatory terms, geolocations, and key technical attributes.  
✔ **Text Summarization** – Generates concise or detailed summaries based on user preference.  
✔ **LLM-Based OCR** – Extracts text from scanned PDFs & DOCX files using OpenAI’s GPT-4o.  
✔ **Performance Logging** – Tracks OpenAI API usage (response time, token count) for cost optimization.  
✔ **Interactive Web UI** – Upload and process documents via a **Gradio-powered interface**.  

---

## 📌 Prerequisites  

Before running this project, ensure you have the following installed:  

### **1️⃣ System Requirements**  
- **OS:** Windows, macOS, or Linux  
- **Python:** `>= 3.8`  
- **Internet Connection:** Required for OpenAI API calls  

### **2️⃣ Required Packages**  
Install dependencies using **pip**:  

```sh
pip install -r requirements.txt


Alternatively, install manually:  
```sh
pip install langchain-openai langchain gradio python-dotenv pdfplumber pymupdf pdf2image python-docx docx2pdf pillow tiktoken
```

### **3️⃣ OpenAI API Key**  
This project uses OpenAI’s **GPT-4o**. You’ll need an **API key**:  

- **Sign up at:** [https://platform.openai.com/signup](https://platform.openai.com/signup)  
- **Get your API key from:** [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)  

Create a `.env` file in the project directory and add:  

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### **4️⃣ Poppler for PDF Processing (Windows Users Only)**  
If you're on **Windows**, install **Poppler** (required for `pdf2image`):  

- **Download:** [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)  
- Extract the ZIP file to `C:\Program Files\poppler`  
- Add `C:\Program Files\poppler\bin` to **System Environment Variables** (`Path`).  
- Verify installation by running:  
  ```sh
  pdftoppm -v
  ```

---

## 📌 How to Run the Project  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/MIVAA-ai/Mivaa-Oil-Gas-Document-Analyzer.git
cd Mivaa-Oil-Gas-Document-Analyzer
```

### **2️⃣ Set Up the Virtual Environment (Optional but Recommended)**  
```sh
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\activate   # Windows
```

### **3️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Run the Application**  
```sh
python mivaa-basic-agentic-ai-text-summary.py
```

### **5️⃣ Access the Web UI**  
Once the application is running, open **Gradio’s interface** in your browser:  
```
http://127.0.0.1:7860
```
Upload a **TXT, PDF, or DOCX file**, and the AI will classify, extract entities, and summarize the document.

---

## 📌 Project Workflow  

1. **File Upload** – Users upload a document via the Gradio UI.  
2. **Text Extraction** – The system extracts text from **TXT, PDF, or DOCX** files.  
3. **Classification** – OpenAI GPT-4o classifies the document based on content.  
4. **Entity Extraction** – Key terms such as well names, geolocations, and reservoir data are identified.  
5. **Summarization** – The document is summarized based on the user’s selected detail level.  
6. **Logging & Performance Tracking** – The system logs response time and token usage for optimization.  

---

## 📌 Troubleshooting  

### **1️⃣ OpenAI API Key Not Found**
Error:  
```
openai.error.AuthenticationError: No API key provided.
```
Solution:  
- Ensure your `.env` file contains `OPENAI_API_KEY`  
- Restart your terminal after updating `.env`  

### **2️⃣ Poppler Not Found (Windows)**
Error:  
```
Error: Unable to get page count. Is poppler installed and in PATH?
```
Solution:  
- Ensure Poppler is installed and added to the system **PATH**.  
- Run `pdftoppm -v` to check installation.  

### **3️⃣ Missing Dependencies**
Error:  
```
ModuleNotFoundError: No module named 'xyz'
```
Solution:  
- Run `pip install -r requirements.txt` to install all dependencies.  

---

## 📌 Contributing  
Contributions are welcome! Feel free to:  
1. **Fork the repository**  
2. **Submit pull requests** for feature improvements  
3. **Report issues** via GitHub  

---

## 📌 License  
This project is released under the **MIT License**.  

---

## 📌 Stay Connected  
For discussions, updates, and collaboration:  
🌐 **Website:** [https://deepdatawithmivaa.com](https://deepdatawithmivaa.com)  
📧 **Contact:** info@deepdatawithmivaa.com  
```