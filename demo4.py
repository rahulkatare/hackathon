# Required Libraries
import streamlit as st
from pathlib import Path

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    SQLStructStoreIndex, 
    SQLDatabase,
    WikipediaReader
)

import pinecone
import os
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import ServiceContext, LLMPredictor
from llama_index.storage import StorageContext
from llama_index.vector_stores import  MilvusVectorStore, PineconeVectorStore
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os, openai
from llama_index.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.vector_store import VectorIndexAutoRetriever

from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

from pathlib import Path
from llama_index import download_loader
# Streamlit Application
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column
from sqlalchemy import insert


def response_to_user_query(user_query, file_path):

    api_key = "c9e62a2b-f08e-461a-a27b-520b9b97b7ba"
    pinecone.init(api_key=api_key, environment="us-east4-gcp")

    # dimensions are for text-embedding-ada-002
    #pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")
    pinecone_index = pinecone.Index("quickstart")
    pinecone_index.delete(deleteAll=True)


    os.environ["OPENAI_API_KEY"] = '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'
    openai.api_key = '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'

    # define node parser and LLM
    chunk_size = 1024
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True))
    service_context = ServiceContext.from_defaults(chunk_size=chunk_size, 
                                                   llm_predictor=llm_predictor)
    text_splitter = TokenTextSplitter(chunk_size=chunk_size)
    node_parser = SimpleNodeParser(text_splitter=text_splitter)

    # define milvus vector index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='wiki_cities')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex([], storage_context=storage_context)

    
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()

    col1 = "Parameter"
    col2 = "Low_Risk"
    col3 = "High_Risk"
    # create city SQL table
    table_name = "parameters"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column(col1, String(100), primary_key=True),
        Column(col2, String(30)),
        Column(col3, String(30)),
    )

    metadata_obj.create_all(engine)

    rows = [
        {"Parameter": "Liquidation-Preference", col2: "", col3: "Participating"},
        {"Parameter": "Dividends", col2: "1% to 10%", col3: ""},
        {"Parameter": "Founder-vesting-period", col2: "5 to 6 years", col3: "> 6 years"},
        {"Parameter": "Anti-dilution", col2: "", col3: "Full Ratchet"},
        {"Parameter": "ESOP", col2: "10 to 15%", col3: "> 15%"},
        {"Parameter": "Exclusivity-Period", col2: "> 120 days", col3: ""},
        {"Parameter": "Legal-Fees", col2: "$30000 to $50000", col3: "> $50000"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.connect() as connection:
            cursor = connection.execute(stmt)
            connection.commit()

    PDFReader = download_loader("PDFReader")

    loader = PDFReader()
    wiki_docs = loader.load_data(file=Path(file_path))

    sql_database = SQLDatabase(engine, include_tables=["parameters"])
    sql_index = SQLStructStoreIndex.from_documents(
        [], 
        sql_database=sql_database, 
        table_name="parameters",
    )

    # Insert documents into vector index
    # Each document has metadata of the city attached
    for wiki_doc in wiki_docs:
        nodes = node_parser.get_nodes_from_documents([wiki_doc])
        vector_index.insert_nodes(nodes)

    sql_query_engine = sql_index.as_query_engine()
    vector_index_query_engine = vector_index.as_query_engine(similarity_top_k=3)

    sql_response = sql_query_engine.query(user_query)
    vector_index_user_query = f"{user_query} considering {str(sql_response)}. Answer only as per the given information."
    final_response = vector_index_query_engine.query(vector_index_user_query)

    return str(final_response)


def main():
    st.title('Auror.AI')
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        #st.write(file_details)
        
        # with open(uploaded_file.name, "wb") as f:
        #     f.write(uploaded_file.getbuffer())
        
        #st.success("File has been stored at local path: {}".format(os.path.join(os.getcwd()+'/new_data/', uploaded_file.name)))
        file_path = os.path.join(os.getcwd()+'/new_data/', uploaded_file.name)

        response = response_to_user_query("Tell me If Dividends is at Low Risk", file_path)
        st.write(response)

        response = response_to_user_query("Tell me If Liquidation-Preference is at High Risk", file_path)
        st.write(response)

        response = response_to_user_query("Tell me If Founder-vesting-period is at High Risk", file_path)
        st.write(response)

        response = response_to_user_query("Tell me If ESOP is at High Risk", file_path)
        st.write(response)

        response = response_to_user_query("Tell me If Anti-dilution is at High Risk", file_path)
        st.write(response)

        response = response_to_user_query("Tell me If Exclusivity-Period is at Low Risk", file_path)
        st.write(response)

if __name__ == '__main__':
    main()



