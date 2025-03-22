from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from qdrant_class import qdrant_ob
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




chunk_size = 500
chunk_overlap = 50


def load_split_pdf_file(file):
    loader = PyMuPDFLoader(file)
    pages = loader.load()
    # pdf_page_content = [page.page_content for page in pages]

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
        #separators= ["\n\n", "\n", " ", ""]
    )

    # Split the text into smaller chunks
    chunks = text_splitter.split_documents(pages)
    return chunks 

chunks = load_split_pdf_file()
embeddings = OpenAIEmbeddings("text-embedding-3-small")
collection_name = "psgpdf"
qdrant_ob.insertion(chunks,embeddings = embeddings,collection_name = collection_name )
qdrant_ob.retrieval(collection_name = collection_name,embeddings = embeddings)
qdrant_ob.create_collection(collection_name = collection_name)

def Conversational_Chain(query, history):
        try:
            template = """you are expert chatbot assistant. you also have user history. Answer questions based on user history.
            history: {HISTORY}
            query:{QUESTION}
            """
            prompt = ChatPromptTemplate.from_template(template)
            model = ChatOpenAI(
                model="gpt-4o-mini", 
                openai_api_key=os.getenv("OPENAI_API_KEY"), 
                temperature=0
                )

            setup = RunnableParallel(
            {"HISTORY": RunnablePassthrough(), "QUESTION": RunnablePassthrough()}
            )

            output_parser = StrOutputParser()

            rag_chain = (
                setup
                | prompt
                | model
                | output_parser
            )
            input_dict = {"QUESTION": query, "HISTORY": history}
            response = rag_chain.invoke(str(input_dict))
            return response
        
        except Exception as e:
            return f"Error executing conversational retrieval chain: {str(e)}"