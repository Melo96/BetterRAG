import re
import json
import streamlit as st
import time
from pathlib import Path
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_transformers import LongContextReorder
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.storage import RedisStore
from FlagEmbedding import FlagReranker
from operator import itemgetter
from prompt_template.prompts import *

from dotenv import load_dotenv
load_dotenv()

llm = "gpt-4-1106-preview"
persist_directory = "/data/kaiwen/knowledge-agent/chroma_bge"
collection_name = "customer_support"
redis_url = "redis://localhost:6379"
top_k = 10
top_k_final = 5
query_count = 4
doc_id_key = "doc_id"

# Load embedding model
@st.cache_resource
def initialize_chain():
    model_name = "BAAI/bge-large-zh-v1.5"
    model_kwargs = {'device': "cuda:0"}
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章: "
    )
    # Load reranker model
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embed_model,
        persist_directory=persist_directory,
    )

    # The storage layer for the parent documents
    docstore = RedisStore(redis_url=redis_url, client_kwargs={"decode_responses": True})
    id_key = "doc_id"

    model = ChatOpenAI(temperature=0, model=llm)
    filter_prompt = ChatPromptTemplate.from_template(FILTER_PROMPT)
    multi_query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    # rag_chain = {"question": itemgetter("question"), "context": lambda x:x} | rag_prompt | model | StrOutputParser()
    filter_chain = {"query": itemgetter("query")} | filter_prompt | model | StrOutputParser()
    multi_query_chain = {"queryCount": itemgetter("queryCount"), "question": itemgetter("question")} | multi_query_prompt | model | StrOutputParser()
    rag_chain = {"questions": lambda x:x, "context": lambda x:x} | rag_prompt | model | StrOutputParser()
    return [vectorstore, docstore, multi_query_chain, filter_chain, rag_chain, reranker]

@st.cache_resource
def rag(ori_query):
    st.text("正在查找答案...")
    # Multi-query generation
    multi_query = st.session_state['multi_query_chain'].invoke({"queryCount": query_count, "question": ori_query})
    queries = [ori_query] + multi_query.split("\n")[1:-1]
    st.markdown(f"""<h5>为您生成相关问题:</h5>
        {str("<br />".join(queries))}
        <br />
        <br />
    """,
    unsafe_allow_html=True
    )

    filters = st.session_state['filter_chain'].invoke({"query": ori_query}).split("|")
    print(filters)
    # Get the matches for each query
    doc_ids = set()
    for query in queries:
        matches = st.session_state['vectorstore'].max_marginal_relevance_search(query, k=top_k, filter={'filter': {"$in": filters}}, lambda_mult=0.5)
        doc_ids.update([match.metadata['doc_id'] for match in matches])
    doc_ids = list(doc_ids)
    matches = st.session_state['docstore'].mget(doc_ids)
    match_list = [json.loads(match) for match in matches]
    match_list_text = [match['page_content'] for match in match_list]

    st.text(f"已为您找到{len(match_list)}个相关来源，正在核对与重排...")
    # Reranking and Reordering
    sentence_pairs = [(ori_query, text) for text in match_list_text]
    scores = st.session_state['reranker'].compute_score(sentence_pairs)
    result_pairs = sorted(zip(match_list, scores), key=lambda x: x[1], reverse=True)
    result_text_pairs = sorted(zip(match_list_text, scores), key=lambda x: x[1], reverse=True)
    result = [pair[0] for pair in result_pairs[:5]]
    result_text = [pair[0] for pair in result_text_pairs[:5]]
    reordering = LongContextReorder()
    result_text = reordering.transform_documents(result_text)
    response = st.session_state['rag_chain'].invoke({"questions": queries, "context": result_text})
    st.markdown(f"""<h3>Agent answers:</h3>
        {str(response)}
        <br />
        <br />
    """,
    unsafe_allow_html=True
    )
    return result, response

# Initialization
if 'files' not in st.session_state:
    st.session_state['files'] = set()

st.title("🦙 Motion G Research Agent 🦙")
retrieval_tab, chat_tab = st.tabs(
        ["Retrieval 知识查找", "Chat about File 文件细节"]
    )

with retrieval_tab:
    st.subheader("知识查找")
    sesstion_state_name = ['vectorstore', 'docstore', 'multi_query_chain', 'filter_chain', 'rag_chain', 'reranker']
    init = initialize_chain()
    for name, func in zip(sesstion_state_name, init):
        st.session_state[name] = func
    ori_query = st.text_input("问题")
    if st.button('搜索') and ori_query:
        st.session_state['ori_query'] = ori_query
        s = time.time()
        result, response = rag(ori_query)
        st.markdown("""<h3>Sources nodes:</h3>""", unsafe_allow_html=True)
        # Get source files
        data_path = Path("/data/kaiwen/data/知识库")
        for i, node in enumerate(result):
            source_file = "/".join([node['metadata']['file_directory'], node['metadata']['file_name']])
            source_file = Path(source_file).relative_to(data_path).as_posix()
            
            text = node['page_content']
            st.markdown(f"""<b>{i+1}. <em>{source_file}, Page {node['metadata']['page_label']}</em></b>""", unsafe_allow_html=True)
            if '<table>' in text:
                st.markdown(f"""{text} <br />""", unsafe_allow_html=True)
            else:
                text = re.sub(r'\s+', ' ', text.replace('\n', ''))
                st.text(f"""{text} <br />""")
            
            st.session_state['files'].add(source_file)

        st.markdown(""" <h3>Download source files:</h3>""", unsafe_allow_html=True)
        for file in list(st.session_state['files']):
            with open(data_path / file, 'rb') as f:
                st.download_button(label=f":arrow_down: {'-'.join(file.split('/'))}", 
                                    data=f.read(),
                                    file_name=f"{'-'.join(file.split('/'))}"
                                )
        e = time.time()
        print("total time:", e-s)

with chat_tab:
    st.subheader("文件细节")

