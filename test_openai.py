import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # For OpenAI chat models
from langchain.schema import HumanMessage

# ✅ Load OpenAI API Key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Ensure API key is loaded
if not openai_api_key:
    raise ValueError("❌ OpenAI API key is missing! Please check your .env file.")

# ✅ Initialize OpenAI Model
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)

# ✅ Test prompt to check response format
test_prompt = "Summarize this: AI is transforming industries."
response = llm.invoke([HumanMessage(content=test_prompt)])

# ✅ Print response type and content
print("🔹 Response Type:", type(response))
print("🔹 Response Content:", response)

# ✅ Handle different response formats
if isinstance(response, str):
    extracted_text = response.strip()
elif hasattr(response, "content"):  # For object-based response
    extracted_text = response.content.strip()
elif isinstance(response, dict) and "content" in response:
    extracted_text = response["content"].strip()
else:
    extracted_text = "⚠️ Error: Unexpected response format."

print("\n✅ Extracted Text:", extracted_text)
