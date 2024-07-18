import os
from dotenv import load_dotenv

# from LangChain tutorial
# import bs4
from langchain import hub

from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")


loader = ...
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = PineconeVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

# retrieve and generate using relevant snippets from docs
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("what is task decomposition?")

# cleanup
vectorstore.delete(delete_all=True)
