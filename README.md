# ChatWithPDF


This repository contains an advanced PDF Chat Assistant built using LangChain, Streamlit, and OpenAI's GPT models. The assistant enables users to upload PDF documents, ask questions, and receive context-aware answers with references to specific pages in the document.

## Features
- **Multi-PDF Support**: Upload multiple PDF files for simultaneous querying.
- **Conversational Interface**: Engage in a conversational flow with memory.
- **Performance Metrics**: Track query performance with visual charts.
- **Customizable Processing Options**: Adjust confidence thresholds for responses.

## Installation

### Prerequisites
- Python 3.8 or higher
- Anaconda (recommended)
- OpenAI API key

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/advanced-pdf-chat-assistant.git
   cd advanced-pdf-chat-assistant
   ```

2. **Create a virtual environment**:
   ```bash
   conda create -n pdf-chat-env python=3.8
   conda activate pdf-chat-env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   ```

5. **Set up environment variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF files** using the sidebar.

3. **Ask questions** in the main chat interface. The assistant will provide answers based on the content of the uploaded PDFs.

4. **View performance metrics** in the dedicated section.

## File Structure
```
.
├── app.py                     # Main Streamlit app
├── components
│   ├── chat_manager.py        # Manages chat logic and chains
│   ├── pdf_processor.py       # Extracts and processes PDF content
│   └── validator.py           # Validates input data
├── utils
│   ├── text_processing.py     # Text processing utilities
│   └── visualization.py       # Performance visualization
├── requirements.txt           # Required Python packages
└── README.md                  # Documentation
```

## Troubleshooting

### Common Errors

1. **NLTK resource not found**:
   Ensure you've downloaded the required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

2. **NameError: name 're' is not defined**:
   Ensure `import re` is present at the top of the file where regular expressions are used.

3. **Streamlit caching issues**:
   If you encounter unexpected behavior, try clearing the cache:
   ```bash
   streamlit run app.py --clear-cache
   ```

## License
This project is licensed under the MIT License.

## Acknowledgments
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI GPT Models](https://openai.com)

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact
For questions or support, please contact sammehta02215@gmail.com.

