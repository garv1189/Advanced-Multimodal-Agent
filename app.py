# Multimodal Agent Application Example
# This script demonstrates how to use the multimodal agent in a practical application

import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
from typing import List, Dict, Any

# Import the multimodal agent implementation
from multimodal_agent import create_multimodal_agent, enhance_agent_with_knowledge_base, process_image_batch

# Configure the page
st.set_page_config(page_title="Multimodal AI Assistant", layout="wide")
st.title("Advanced Multimodal AI Assistant")

# Initialize the agent
@st.cache_resource
def initialize_agent():
    """Initialize the multimodal agent."""
    agent = create_multimodal_agent()
    
    # If you have a knowledge base folder, uncomment the following line
    # agent = enhance_agent_with_knowledge_base(agent, "knowledge_base")
    
    return agent

agent = initialize_agent()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_url"):
            st.image(message["image_url"], caption="Uploaded Image")
        st.markdown(message["content"])

# Function to handle image uploads and URLs
def process_image(image_source):
    """Process an image from a file upload or URL."""
    image_url = None
    
    if isinstance(image_source, str) and image_source.startswith("http"):
        # Handle image URL
        try:
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content))
            image_url = image_source
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")
            return None
    else:
        # Handle uploaded image
        try:
            image = Image.open(image_source)
            # Save to a temporary file and get URL (in a real app, you'd upload to a service)
            temp_path = f"temp_image_{int(time.time())}.jpg"
            image.save(temp_path)
            image_url = temp_path  # In a real app, this would be a full URL
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")
            return None
    
    return image_url

# Chat input area
with st.container():
    # Allow user to input text or upload an image
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.chat_input("Ask something or provide context...")
        
    with col2:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Process image if uploaded
    image_url = None
    if uploaded_file:
        image_url = process_image(uploaded_file)
        st.image(image_url, caption="Uploaded Image", width=300)
    
    # Handle user input
    if user_input:
        # Add URL to message if image was uploaded
        full_message = user_input
        if image_url:
            full_message += f" [Image: {image_url}]"
        
        # Display user message
        st.chat_message("user").markdown(user_input)
        if image_url:
            st.chat_message("user").image(image_url, caption="Uploaded Image")
        
        # Add to session state
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "image_url": image_url
        })
        
        # Get agent response with loading indicator
        with st.spinner("Thinking..."):
            response = agent(full_message)
        
        # Display assistant response
        st.chat_message("assistant").markdown(response)
        
        # Add to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })

# Sidebar with advanced options
with st.sidebar:
    st.header("Advanced Options")
    
    # Batch processing option
    st.subheader("Batch Image Processing")
    batch_urls = st.text_area("Enter image URLs (one per line)")
    
    if st.button("Process Batch"):
        if batch_urls:
            urls = [url.strip() for url in batch_urls.split("\n") if url.strip()]
            
            if urls:
                with st.spinner("Processing batch of images..."):
                    results = process_image_batch(agent, urls)
                
                # Display results
                for i, result in enumerate(results):
                    st.subheader(f"Image {i+1}")
                    st.image(result["url"], width=200)
                    st.write(result["analysis"])
            else:
                st.warning("No valid URLs provided")
    
    # Knowledge base management (placeholder for full implementation)
    st.subheader("Knowledge Base")
    st.file_uploader("Upload documents to knowledge base", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    st.caption("Note: Knowledge base functionality requires additional setup")

# Additional app features
st.header("Features")
st.markdown("""
This multimodal assistant can:
- Analyze images and describe their content
- Extract text from images (OCR)
- Answer questions about uploaded images
- Search through knowledge bases
- Process documents and retrieve relevant information
- Generate detailed descriptions of concepts
- Handle conversations with mixed text and image inputs
""")

# Footer
st.caption("Powered by LangChain and OpenAI")
