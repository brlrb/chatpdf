
import os
import chainlit as cl
from chainlit.types import AskFileResponse


from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

import tiktoken

from dotenv import load_dotenv


load_dotenv()


# START CODE
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Helper function

def tiktoken_func(file):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        file,
    )
    return len(tokens)


async def process_file(file: AskFileResponse):

    print("process_file - filefilefile: ", file)

    pypdf_loader = PyMuPDFLoader(file.path)
    texts = pypdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=150,
        length_function=tiktoken_func
    )

    documents = text_splitter.split_documents(texts)

    return documents


async def RAG_pipeline(question: str, documents: str):

    template = """
    
    Answer the question based only on the following context. 
    
    If you cannot answer the question with the context, please respond with 'I don't know, can you provide more context?'.

    If the user question is not related to the uploaded document then respond with "My apologies, human. I can only help you with the document you uploaded.\n\n".

    Always answer in full sentence.

    If a user says things like, "Hi", "Hello", or anything related to greetings then respond with a nice kind greetings in British english accent.
    
    If a user says things like "Ok" or "thank you" or "thank you" or anything that is related to  "phatic expressions" or "phatic communication" then respond with "No problem! Always happy to help."
    
    If you provided answer based on the question and context then ONLY end your sentence on a New Line with " \n\n\n\n Thank you for asking. What else can I answer about the document?".

    If the user asks question such as, "Hi Bikram", "Hi BK", or greetings with the word, "Bk", "Bikram" then respond with "Lets get serious Budd Chris :)"


    Context:
    {context}

    Question:
    {input}
    """

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Initialize a retriever to retrieve similar context
    retriever = vector_store.as_retriever()

    # Initialize retriever using a multi-query approach with a language model.
    retriever = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=openai_chat_model)

    # Create a document chain using OpenAI chat model and a prompt
    document_chain = create_stuff_documents_chain(openai_chat_model, prompt)

    # Create a retrieval chain using a retriever and a document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Send a request to OpenAI with the question
    response = retrieval_chain.invoke({"input": question})

    # Making sure we have 'answer' params so that we can give proper response
    if 'answer' in response:
        llm_answer = response['answer']
    else:
        llm_answer = '**EMPTY RESPONSE**'

    print("llm_answer: ", llm_answer)

    return llm_answer


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    print("A new chat session has started!")

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!", accept=["application/pdf"]
        ).send()

    file = files[0]

    # Let the user know that the system is ready
    msg = cl.Message(
        content=f"Processing `{file.name}`..."
    )
    await msg.send()

    pdf_file = await process_file(file)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` uploaded. You can now ask questions!"
    await msg.update()

    # Save session
    cl.user_session.set("documents", pdf_file)
    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):

    print("Human asked: ", message.content)

    msg = cl.Message(content="")
    await msg.send()

    # do some work
    await cl.sleep(2)

    # Retrieve session
    document_chunks = cl.user_session.get("documents")

    # Wait for OpenAI to return a response and the good ol' RAG stuff
    response = await RAG_pipeline(message.content, document_chunks)

    # If there is a response then let the user know else fallback to else statement!
    if response:
        await cl.Message(content=response).send()
    else:
        cl.Message(
            content="Something went wrong! please kindly refresh and try again 🤝").send()
