# Advanced Multimodal Agent with LangChain

This project implements an advanced multimodal agent capable of processing and reasoning across both text and images using LangChain. The agent can analyze images, extract text from images, search through knowledge bases, and generate insightful responses that combine visual and textual understanding.

## Features

- **Multimodal Understanding**: Process both text and images seamlessly
- **Image Analysis**: Analyze image content and provide detailed descriptions
- **Optical Character Recognition (OCR)**: Extract text from images
- **Knowledge Base Integration**: Search through vector databases for relevant information
- **Enhanced Memory System**: Track conversation history and image references
- **Streamlit Web Interface**: User-friendly interface for interacting with the agent
- **Batch Processing**: Process multiple images at once

## Prerequisites

- Python 3.8+
- OpenAI API key (for GPT-4V and embeddings)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/multimodal-agent.git
   cd multimodal-agent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

```
multimodal-agent/
├── multimodal_agent.py      # Main agent implementation
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
├── knowledge_base/          # Folder for knowledge base documents
└── README.md                # This documentation
```

## Usage

### Running the Web Interface

```bash
streamlit run app.py
```

This will start the Streamlit web interface where you can interact with the agent through text and image inputs.

### Using the Agent Programmatically

```python
from multimodal_agent import create_multimodal_agent

# Create the agent
agent = create_multimodal_agent()

# Ask questions with text
response = agent("What is the capital of France?")
print(response)

# Ask questions about an image
response = agent("What can you see in this image? https://example.com/image.jpg")
print(response)

# Ask follow-up questions
response = agent("Based on the image I just showed you, what time of day is it?")
print(response)
```

### Adding a Knowledge Base

To enhance the agent with a custom knowledge base:

1. Create a folder called `knowledge_base` and add your PDF, TXT, or DOCX files.
2. Use the `enhance_agent_with_knowledge_base` function:

```python
from multimodal_agent import create_multimodal_agent, enhance_agent_with_knowledge_base

agent = create_multimodal_agent()
enhanced_agent = enhance_agent_with_knowledge_base(agent, "path/to/knowledge_base")

# Now the agent can search through your documents
response = enhanced_agent("What does my document say about multimodal AI?")
print(response)
```

## Tools and Capabilities

The agent comes with several built-in tools:

1. **Image Analysis Tool**: Analyzes image content and provides detailed descriptions
2. **OCR Tool**: Extracts text from images
3. **Knowledge Base Search**: Searches through vector databases for relevant information
4. **Description Generator**: Generates detailed descriptions of concepts

## Extending the Agent

### Adding New Tools

You can easily extend the agent with custom tools:

```python
from langchain.tools import BaseTool

class MyCustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of what this tool does"
    
    def _run(self, input_parameter: str) -> str:
        # Implementation of the tool
        return f"Result of processing {input_parameter}"

# Add the tool to the agent
tools.append(MyCustomTool())
```

### Customizing the System Prompt

To modify the agent's behavior, you can customize the system prompt in `create_multimodal_agent()`.

## Performance Considerations

- The agent uses GPT-4V for image analysis, which may have usage costs.
- Processing large images or documents may take time.
- For batch processing of many images, consider implementing rate limiting.

## Future Enhancements

- Add support for audio and video processing
- Implement more sophisticated memory systems
- Add document chunking for better knowledge retrieval
- Support for more document formats
- Implement caching for improved performance

## Troubleshooting

**Issue**: Image URL not loading
**Solution**: Ensure the image URL is publicly accessible and the format is supported (JPG, PNG, etc.)

**Issue**: Knowledge base not returning results
**Solution**: Check that your documents are properly formatted and that the embedding model is correctly initialized.

**Issue**: Rate limits or API errors
**Solution**: Implement rate limiting or backoff mechanisms for API calls.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the framework
- OpenAI for GPT-4V and embedding models
