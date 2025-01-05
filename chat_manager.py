from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

class ChatManager:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> PromptTemplate:
        """Create a strict prompt template for accurate responses."""
        template = """You are a precise and accurate AI assistant that helps users understand their PDF documents.
        
        Important Instructions:
        1. Only use the provided context to answer the question
        2. If the context doesn't contain enough information to answer fully, explicitly state what you cannot answer
        3. If you're unsure about any part of the answer, express your uncertainty
        4. Include relevant page numbers in your response when available
        5. Do not make assumptions or inferences beyond what's directly supported by the context
        6. If multiple contexts provide conflicting information, point out the discrepancy
        
        Context: {context}
        
        Question: {question}
        
        If you cannot answer the question based solely on the provided context, say:
        "I apologize, but I cannot provide a complete answer based on the available context. Would you like me to explain what information is missing?"
        
        Otherwise, provide your response with specific references to the context:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def create_chain(self, vectorstore) -> ConversationalRetrievalChain:
        """Create a conversation chain with the configured properties."""
        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        condense_question_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone Question:"""
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': self.prompt},
            return_source_documents=True,
            condense_question_prompt=condense_question_prompt,
            verbose=True
        )
    
    def process_response(
        self,
        response: Dict,
        context_scores: List[Dict[str, float]]
    ) -> Dict:
        """Process and enhance the response with metadata."""
        return {
            'answer': response['answer'],
            'source_documents': response.get('source_documents', []),
            'context_scores': context_scores,
            'confidence_score': self._calculate_confidence(context_scores)
        }
    
    def _calculate_confidence(self, context_scores: List[Dict[str, float]]) -> float:
        """Calculate overall confidence score for the response."""
        if not context_scores:
            return 0.0
        
        # Use the highest scoring context as the confidence indicator
        max_score = max(score['final_score'] for score in context_scores)
        return max_score
