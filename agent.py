import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize vision model and embedding model
vision_model = ChatOpenAI(model="gpt-4-vision-preview", temperature=0)
embedding_model = OpenAIEmbeddings()

# Custom tools for the multimodal agent
class ImageAnalysisTool(BaseTool):
    name = "image_analysis"
    description = "Analyzes image content and provides detailed descriptions"
    
    def _run(self, image_url: str) -> str:
        """Analyze the content of an image from a URL."""
        try:
            # Download the image
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            
            # Create a message with the image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image in detail, including all visible elements, text, colors, objects, and overall scene."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )
            
            # Get response from vision model
            result = vision_model.invoke([message])
            return result.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

class OCRTool(BaseTool):
    name = "extract_text_from_image"
    description = "Extracts and recognizes text from images"
    
    def _run(self, image_url: str) -> str:
        """Extract text from an image URL."""
        try:
            # Create a message focusing on text extraction
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Extract all text content visible in this image. Return only the text, formatted properly."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )
            
            # Get response from vision model
            result = vision_model.invoke([message])
            return result.content
        except Exception as e:
            return f"Error extracting text: {str(e)}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search through the agent's knowledge base for relevant information."""
    # In a real implementation, this would connect to a vector store
    # For now, returning a placeholder response
    return f"Searching knowledge base for: {query}. In a full implementation, this would return relevant information from a vector database."

@tool
def generate_image_description(concept: str) -> str:
    """Generate a detailed description of what an image of the given concept might look like."""
    prompt = f"Generate a detailed visual description of {concept}."
    response = ChatOpenAI(temperature=0.7).invoke([HumanMessage(content=prompt)])
    return response.content

class MultimodalMemory:
    """Enhanced memory system that stores both text interactions and references to images."""
    
    def __init__(self):
        self.conversation_memory = ConversationBufferMemory(return_messages=True)
        self.image_references = []
        
    def add_user_message(self, message: str):
        self.conversation_memory.chat_memory.add_user_message(message)
        
    def add_ai_message(self, message: str):
        self.conversation_memory.chat_memory.add_ai_message(message)
        
    def add_image_reference(self, image_url: str, description: str):
        self.image_references.append({
            "url": image_url,
            "description": description,
            "timestamp": import datetime; datetime.datetime.now().isoformat()
        })
        
    def get_conversation_history(self):
        return self.conversation_memory.chat_memory.messages
        
    def get_image_context(self):
        """Return context about images that have been processed."""
        if not self.image_references:
            return "No images have been processed yet."
            
        context = "Previously analyzed images:\n"
        for i, img in enumerate(self.image_references[-3:]):  # Last 3 images
            context += f"{i+1}. {img['description']} (URL: {img['url']})\n"
        return context

def create_multimodal_agent():
    """Create and configure the multimodal agent."""
    
    # Initialize tools
    image_analysis_tool = ImageAnalysisTool()
    ocr_tool = OCRTool()
    
    # Combine all tools
    tools = [
        image_analysis_tool,
        ocr_tool,
        search_knowledge_base,
        generate_image_description
    ]
    
    # Create a memory system
    memory = MultimodalMemory()
    
    # Create a comprehensive system prompt
    system_prompt = """You are an advanced multimodal AI assistant capable of understanding and reasoning about both text and images.

You can analyze images, extract text from images, search through knowledge bases, and generate descriptions.

When processing images:
1. Describe what you see in detail
2. Extract any visible text when relevant
3. Connect visual information with contextual knowledge
4. Consider how the image relates to the user's query

Always think step-by-step about the best approach to solve the user's request considering all available modalities.
If you need to see an image, ask the user to provide a URL.

Your goal is to provide helpful, accurate, and insightful responses that leverage both visual and textual understanding.
"""
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    # Define the full agent chain with memory
    def invoke_agent(user_input):
        # Add user message to memory
        memory.add_user_message(user_input)
        
        # Get chat history
        chat_history = memory.get_conversation_history()
        
        # Execute agent
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "agent_scratchpad": []
        })
        
        # Add AI response to memory
        memory.add_ai_message(response["output"])
        
        return response["output"]
    
    return invoke_agent

# Example usage
if __name__ == "__main__":
    multimodal_agent = create_multimodal_agent()
    
    # Example interactions
    print(multimodal_agent("Can you tell me what this image shows? https://example.com/sample_image.jpg"))
    print(multimodal_agent("Can you extract the text from this document? https://example.com/document.jpg"))
    print(multimodal_agent("Based on our previous discussion about the image, what can you tell me about similar objects?"))

# Additional utility functions for a complete implementation

def setup_knowledge_base(documents_folder: str):
    """Set up a vector store knowledge base from a folder of documents."""
    documents = []
    
    # Process PDF files
    for filename in os.listdir(documents_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(documents_folder, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embedding_model)
    retriever = vectorstore.as_retriever()
    
    return retriever

def enhance_agent_with_knowledge_base(agent_fn, knowledge_base_folder: str):
    """Enhance the agent with a knowledge base."""
    retriever = setup_knowledge_base(knowledge_base_folder)
    retriever_tool = create_retriever_tool(
        retriever,
        "search_documents",
        "Search for information in the uploaded documents and knowledge base."
    )
    
    # In a real implementation, we would modify the agent to include this tool
    
    return agent_fn

def process_image_batch(agent_fn, image_urls: List[str]):
    """Process a batch of images with the agent."""
    results = []
    for url in image_urls:
        result = agent_fn(f"Analyze this image in detail: {url}")
        results.append({"url": url, "analysis": result})
    return results
