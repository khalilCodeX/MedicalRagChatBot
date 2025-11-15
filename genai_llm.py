from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import management
import dataloader
import vectordb
from token_calc import calculate_tokens, calculate_price
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class genai_llm:
    def __init__(self):
        self.openai_client = management.get_openai_client()
        self.data_loader = dataloader.dataLoader()
        self.qdrant_client = management.get_qdrant_client()
        self.vector_db = vectordb.VectorDB(self.qdrant_client)
        self.chat_history = []
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)

    @property
    def get_data_loader(self):
        return self.data_loader
    
    def preprocess_data(self):
        head = self.data_loader.view_head()
        print(f"DataFrame Head:\n{head}")
        
        shape = self.data_loader.get_shape()
        print(f"DataFrame Shape: {shape}")

        self.data_loader.add_document_col()
        head = self.data_loader.view_head()
        print(f"DataFrame Head after adding 'document' column:\n{head}")
    
    def tokenize_chunkify_documents(self):
        docs = self.data_loader.get_documents()
        doc_chunks = self.data_loader.chunk_doc(docs)
        print(f"Number of document chunks: {len(doc_chunks)}")
        return doc_chunks
    
    def embed_documents(self, doc_chunks):
        if doc_chunks:
            self.vector_db.embed_text(doc_chunks)

    def set_embedding_vector(self):
        self.vector_db.set_embedding_vector()

    def calculate_costs(self):
        full_text = "\n\n".join(self.data_loader.df['document'].tolist())
        token_len = calculate_tokens(full_text, model="gpt-5-nano")
        print(f"Total number of tokens in 'document' column: {token_len}")

        price = calculate_price(token_len, model="gpt-5-nano")
        print(f"Estimated price for processing tokens: ${price:.2f}")

        embedding_price = calculate_price(token_len, model="text-embedding-3-small", price_per_1M_input_tokens=0.02, price_per_1M_output_tokens=0.0)
        print(f"Estimated price for processing tokens: ${embedding_price:.2f}")
    
    def format_prompt_context(self, siilar_doc):
        formatted_context = []
        for i, doc in enumerate(siilar_doc, start=1):
            doc_text = f"--- Medical Case {i} ---\n"
            doc_text += f"Description: {doc.page_content}\n"
            
            # Add metadata if available
            if doc.metadata:
                if doc.metadata.get('Patient'):
                    doc_text += f"Patient Query: {doc.metadata['Patient']}\n"
                if doc.metadata.get('Doctor'):
                    doc_text += f"Doctor Response: {doc.metadata['Doctor']}\n"
        
            formatted_context.append(doc_text)
    
        context = "\n\n".join(formatted_context)
        return context

    def create_chain(self, user_prompt):
        system_prompt = (
            """You are an expert medical assistant AI with access to a database of medical cases and doctor responses.

Your role is to:
1. Provide accurate medical information based on the context provided
2. Reference specific cases from the database when relevant
3. Maintain a professional and empathetic tone
4. Clearly state when information is not available in the provided context
5. Never provide medical diagnoses - only share information and suggest consulting healthcare professionals

When answering:
- Use the provided medical cases as reference material
- Cite relevant patient queries and doctor responses when applicable
- Be clear, concise, and helpful
- If the context doesn't contain relevant information, acknowledge this limitation

IMPORTANT: Always remind users that this information is for educational purposes and they should consult with qualified healthcare professionals for medical advice.

Context from similar medical cases:
{context}
"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{user_prompt}"),
            ]
        )

        similar_doc = self.vector_db.retrieve_similar_docs(user_prompt)
        context = self.format_prompt_context(similar_doc)

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "context": context,
                "user_prompt": user_prompt,
                "chat_history": self.chat_history,
            }
        )

        return response