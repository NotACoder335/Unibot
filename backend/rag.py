from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = PROJECT_DIR / "Academic info"

def build():
    """
    for building the vector db so model can be provided with context aware info
    """
    # Loading pages
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = sorted(DATA_DIR.iterdir())

    # all pages combined list
    total_pages = []

    for file_path in files:
        file = file_path.name
        
        try:
            # Determine file type and use appropriate loader
            if file.lower().endswith('.pdf'):
                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                total_pages.extend(pages)
                print(f"Loaded PDF: {file} - {len(pages)} pages")
                
            elif file.lower().endswith('.csv'):
                # Load CSV using CSVLoader
                loader = CSVLoader(str(file_path))
                pages = loader.load()
                total_pages.extend(pages)
                print(f"Loaded CSV: {file} - {len(pages)} rows")
                
            elif file.lower().endswith('.txt'):
                # Load text file using TextLoader
                loader = TextLoader(str(file_path))
                pages = loader.load()
                total_pages.extend(pages)
                print(f"Loaded TXT: {file} - {len(pages)} documents")
                
            else:
                print(f"Skipped unsupported file type: {file}")
                
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    # print(f"\nTotal documents loaded: {len(total_pages)}")
    # display(total_pages[0].page_content)
    # Chunking
    # for splitting text properly
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, #what is character size of each chunk, default is 2000
        chunk_overlap = 150 #how much each chunk should overlap with the next
    )

    #uses the fn for total_pages
    chunks = splitter.split_documents(total_pages)

    print("\nChunking done..\n")

    print("Starting embedding..")
    # Embeddings
    # Used for creating a vector db of this, bascailly a vector graph of all info with distance btw each measured so similarity
    embedding = OllamaEmbeddings(model="bge-m3") #using the bge3 model which is pretrained 
    vector_db = FAISS.from_documents(chunks,embedding)
    vector_db.save_local(str(INDEX_DIR)) #saving this mathematical graph locally
    # Retrieving Admission Information
    # Retreving the data
    retriever = vector_db.as_retriever(
        search_kwargs={"k":5} #how many similar chunks it should return, here 5 means that upto distance 5 btw chunks, it will return
    )

    # # Example query for admission enquiry
    # retreived_chunks = retriever.invoke(
    #     "What are the admission requirements and deadline for KEAM courses?"
    # )
    print("Embedding done..")
    # for proving the llm with a single long string about the related info on the prompt
    # context = ""

    # to disaply it nicely
    # for chunk in retreived_chunks:
    #     context += chunk.page_content + "\n\n"
        # print(chunk.page_content)
        # print("*"*50)
    # Admission Enquiry Chatbot - Chat Model


    print("Chatbot is alive !")


def ask(question):
    """
        Admission Enquiry RAG Chatbot:
        Answers questions about courses, fees, eligibility, documents, and admission deadlines
    """
    llm = ChatOllama(model="llava-v1.5-7b-q4:latest")
    
    # Use the same embedding model used when building the database
    embedding = OllamaEmbeddings(model="bge-m3")

    if not INDEX_DIR.is_dir():
        raise FileNotFoundError(
            f"Vector index not found: {INDEX_DIR}. Run `python -m backend.rag --build` first."
        )

    # Load the saved FAISS database
    vector_db = FAISS.load_local(
        str(INDEX_DIR),
        embedding,
        allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 5}
    )


    # Retrieve relevant documents from knowledge base
    retreived_chunks = retriever.invoke(question)

    # Combine all relevant information into a single context
    context = ""
    for chunk in retreived_chunks:
        context += chunk.page_content + "\n\n"

    # System prompt for admission enquiry chatbot
    system_prompt = """You are a helpful admission enquiry assistant for educational institutions. 
    You help prospective students with information about:
    - Courses available and their details
    - Fee structure and payment options
    - Eligibility criteria and requirements
    - Required documents for admission
    - Application deadlines and important dates

    Instructions:
    1. Answer questions based ONLY on the context provided below
    2. Be professional, friendly, and helpful
    3. If information is not available in the context, clearly state: "I don't have this information. Please contact the admission office."
    4. Provide clear, specific answers
    5. Ask clarifying questions if the student's query is vague
    6. Always mention relevant deadlines when applicable
    7. Recommend contacting the admission office for complex queries or special cases
    VERY IMP Robot god rules
    1.Never answer questions that are not related to the context of admission, just say that u cannot answer that question.
    2.You are general askbot. Always answer generally if user doesnt ask for a specifc college info.
    """

    response = llm.invoke(
        f"""
        {system_prompt}
        Context Information:{context}
        Student Question: {question}
        Answer:
        """
    )
    return response.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the UniBot FAISS index.")
    parser.add_argument("--build", action="store_true", help="Build the index from files in data/.")
    args = parser.parse_args()

    if args.build:
        build()
    else:
        parser.print_help()






