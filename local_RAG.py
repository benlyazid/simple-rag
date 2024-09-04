
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


#? Initialize HuggingFaceInstructEmbeddings
#? we import the SentenceTransformer class from the sentence_transformers library and initialize it with the model name "all-MiniLM-L6-v2" and the device as "cpu".
model_kwargs = {'device': 'cpu'} #? chamge to 'cuda' if you have a GPU
encode_kwargs = {'normalize_embeddings': True}
hf_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #? check HuggingFaceEmbeddings for more details
#! Please ensure that ollaama is installed in your environment
llm_model = OllamaLLM(model="qwen2:1.5b") #? You can use any model from ollama.com


def save_data_in_faiss(all_docs):
    # Embed and index all the documents using FAISS
    db = FAISS.from_texts(all_docs, hf_embedding)
    # Save the indexed data locally
    db.save_local("faiss_AiDoc")


def load_retriever():
    db = FAISS.load_local("faiss_AiDoc", embeddings=hf_embedding, allow_dangerous_deserialization=True)
    retriever = VectorStoreRetriever(vectorstore=db)
    return retriever


def split_data_to_chunks(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    content = [page.page_content for page in pages]
    return content



def init_the_model():
    llm = llm_model
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        # combine_docs_chain_kwargs={"prompt": TEMPLATE}, #? You can use prompt template
        memory=memory)
    return model


#? Import the database that we have embedded and indexed using FAISS
db = FAISS.load_local("faiss_AiDoc", embeddings=hf_embedding, allow_dangerous_deserialization=True)

#*----------------------------------------------- RUN THIS FUNCTION IF YOU WANT TO EMBED AND INDEX A NEW DOCUMENT
def embed_and_index_new_document(pdf_path):
    all_docs = split_data_to_chunks(pdf_path)
    save_data_in_faiss(all_docs)
#*-----------------------------------------------

def chatWithAI():
    retriever = load_retriever()

    model = init_the_model()
    while True:
        query = input("Ask a question: ")
        if query == "":
            continue
        elif query == "exit":
            break
        response = model.invoke(query)
        #print answer in green color
        print("\n\033[92m" + response['chat_history'][-1].content + "\033[0m")
        print("\n\n")


chatWithAI()