#this commented code is using openai api key and the later one is using open source model 
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.chat_models import ChatOpenAI

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Import our components
# from components.pdf_processor import PDFProcessor
# from components.validator import AdvancedValidator
# from components.chat_manager import ChatManager
# from components.metrics import PerformanceMetrics, QueryTimer
# from utils.visualization import PerformanceVisualizer
# import nltk
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# # Page configuration
# st.set_page_config(
#     page_title="Advanced PDF Chat Assistant",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# def init_session_state():
#     if 'conversation' not in st.session_state:
#         st.session_state.conversation = None
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'metrics' not in st.session_state:
#         st.session_state.metrics = PerformanceMetrics()
#     if 'documents' not in st.session_state:
#         st.session_state.documents = []
#     if 'page_map' not in st.session_state:
#         st.session_state.page_map = {}


# def main():
#     init_session_state()
    
#     st.title("📚 Advanced PDF Chat Assistant")
    
#     # Initialize components
#     embeddings = OpenAIEmbeddings()
#     pdf_processor = PDFProcessor()
#     validator = AdvancedValidator(embeddings)
#     chat_manager = ChatManager()
#     visualizer = PerformanceVisualizer()
    
#     # Sidebar for document upload and processing
#     with st.sidebar:
#         st.header("📄 Document Upload")
#         pdf_files = st.file_uploader(
#             "Upload your PDF files",
#             type="pdf",
#             accept_multiple_files=True
#         )
        
#         # Processing options
#         st.header("⚙️ Processing Options")
#         confidence_threshold = st.slider(
#             "Confidence Threshold",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.6,
#             step=0.05
#         )
        
#         if st.button("Process Documents"):
#             if pdf_files:
#                 with st.spinner("Processing documents..."):
#                     all_documents = []
#                     page_map = {}
                    
#                     for pdf in pdf_files:
#                         st.write(f"Processing {pdf.name}...")
#                         documents, doc_page_map = pdf_processor.process_pdf(pdf)
#                         all_documents.extend(documents)
#                         page_map.update(doc_page_map)
                    
#                     st.session_state.documents = all_documents
#                     st.session_state.page_map = page_map
                    
#                     st.success("Documents processed successfully.")
                    
#                     # Create vector store
#                     vectorstore = FAISS.from_documents(all_documents, embeddings)
#                     st.session_state.conversation = chat_manager.create_chain(vectorstore)
#                     st.write("Conversation chain created successfully.")

#     # Main chat interface
#     st.header("🧵 Ask Your Questions")
#     user_question = st.text_input("Type your question here")
    
#     if st.session_state.conversation is None:
#         st.warning("Please process documents first to start a conversation.")
#     elif user_question:
#         with st.spinner("Generating response..."):
#             response = st.session_state.conversation({'question': user_question})
#             st.session_state.chat_history = response['chat_history']
            
#             for i, message in enumerate(st.session_state.chat_history):
#                 if i % 2 == 0:
#                     st.write(f"**User**: {message.content}")
#                 else:
#                     st.write(f"**Bot**: {message.content}")

#     # Performance visualization
#     st.header("📊 Performance Metrics")
    
#     if st.session_state.metrics.total_queries > 0:
#         metrics_summary_fig = visualizer.create_metrics_summary(st.session_state.metrics.get_metrics_summary())
#         st.plotly_chart(metrics_summary_fig)

#         radar_chart_fig = visualizer.create_score_radar_chart(st.session_state.metrics.get_average_metrics())
#         st.plotly_chart(radar_chart_fig)

#         metrics_timeline_fig = visualizer.create_metrics_timeline(st.session_state.metrics.get_metrics_history())
#         st.plotly_chart(metrics_timeline_fig)


# if __name__ == '__main__':
#     main()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from components.pdf_processor import PDFProcessor
from components.validator import AdvancedValidator
from components.chat_manager import ChatManager
from components.metrics import PerformanceMetrics, QueryTimer
from utils.visualization import PerformanceVisualizer
import os
from dotenv import load_dotenv
import nltk
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced PDF Chat Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = PerformanceMetrics()
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'page_map' not in st.session_state:
        st.session_state.page_map = {}
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


class MockChatLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        return "This is a mock response for testing purposes."

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "mock_llm"


# Previous imports remain the same...

def main():
    init_session_state()

    st.title("📚 Advanced PDF Chat Assistant")

    # Initialize components
    embeddings = st.session_state.embeddings
    pdf_processor = PDFProcessor()
    validator = AdvancedValidator(embeddings)
    chat_manager = ChatManager()
    visualizer = PerformanceVisualizer()
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("📄 Document Upload")

        pdf_files = st.file_uploader(
            "Upload your PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        # Processing options
        st.header("⚙️ Processing Options")

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05
        )

        if st.button("Process Documents"):
            if pdf_files:
                with st.spinner("Processing documents..."):
                    all_documents = []
                    page_map = {}

                    for pdf in pdf_files:
                        st.write(f"Processing {pdf.name}...")
                        documents, doc_page_map = pdf_processor.process_pdf(pdf)
                        all_documents.extend(documents)
                        page_map.update(doc_page_map)

                    st.session_state.documents = all_documents
                    st.session_state.page_map = page_map

                    st.success("Documents processed successfully.")

                    try:
                        # Create vector store and configure retriever
                        vectorstore = FAISS.from_documents(all_documents, embeddings)
                        st.session_state.conversation = chat_manager.create_chain(vectorstore)
                        st.success("Conversation chain created successfully.")
                    except Exception as e:
                        st.error(f"Error creating conversation chain: {str(e)}")

    # Main chat interface
    st.header("🧵 Ask Your Questions")
    user_question = st.text_input("Type your question here")

    if st.session_state.conversation is None:
        st.warning("Please process documents first to start a conversation.")
    elif user_question:
        with st.spinner("Generating response..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"**User**: {message.content}")
                else:
                    st.write(f"**Bot**: {message.content}")

    # Performance visualization
    st.header("📊 Performance Metrics")

    if st.session_state.metrics.total_queries > 0:
        metrics_summary_fig = visualizer.create_metrics_summary(st.session_state.metrics.get_metrics_summary())
        st.plotly_chart(metrics_summary_fig)

        radar_chart_fig = visualizer.create_score_radar_chart(st.session_state.metrics.get_average_metrics())
        st.plotly_chart(radar_chart_fig)

        metrics_timeline_fig = visualizer.create_metrics_timeline(st.session_state.metrics.get_metrics_history())
        st.plotly_chart(metrics_timeline_fig)


if __name__ == '__main__':
    main()
