import os
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class dataLoader:
    file = 'ai-medical-chatbot.csv'
    
    def __init__(self):  
        if os.path.exists(self.file):
            self.df = pd.read_csv(self.file)
        else:
            self.df = None

    def view_head(self):
        if self.df is not None:
            return self.df.head()
        return None
    
    def get_shape(self):
        if self.df is not None:
            return self.df.shape
        return None

    def add_document_col(self):
        if self.df is not None:
            self.df['document'] = (self.df['Description'].fillna('') + '\n\n' + "Patient: " + self.df['Patient'].fillna('') + 
            '\n\n' + "Doctor: " + self.df["Doctor"].fillna(''))
    
    def get_documents(self):
        if self.df is not None and self.df['Description'].notnull().any():
            self.df['Patient'] = self.df['Patient'].fillna('')
            self.df['Doctor'] = self.df['Doctor'].fillna('')

            docs = [
                Document(
                    page_content=row['Description'],
                    metadata = {"Patient": row['Patient'], 
                                "Doctor": row['Doctor']
                            }
            ) for _, row in self.df.iterrows()]
            return docs
        return []

    def chunk_doc(self, documents, chunk_size=1000, overlap=200):
       text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
       )
       return text_splitter.split_documents(documents)