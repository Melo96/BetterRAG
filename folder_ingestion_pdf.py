import json
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.storage import RedisStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Optional
from prompt_template.prompts import SUMMARY_PROMPT

from dotenv import load_dotenv
load_dotenv()

partition_path = Path("/data/kaiwen/knowledge-agent/pdf_partition")
partitions = list(partition_path.rglob("*.json"))

partition_list = []
for p in partitions:
    file_name = str(p.relative_to(partition_path))
    with open(p, 'r') as f:
        partition_elements = json.load(f)
    partition_list.append((file_name, partition_elements))

class Element(BaseModel):
    type: str
    text: Any
    page_label: Optional[int] = None
    file_name: str
    id: str

# Categorize by type
categorized_elements = []
table_pages = []
text_pages = []
for file_name, partition_elements in partition_list:
    page_num = 1
    for element in partition_elements:
        if element['type']=="PageBreak" or element['text']=="":
            categorized_elements.append(Element(type="page_break", text=page_num, file_name=file_name, id=element['id']))
            page_num += 1
            continue
        if element['type']=="Table":
            categorized_elements.append(Element(type="table", text=element['text'], page_label=page_num, file_name=file_name, id=element['id']))
            table_pages.append(page_num)
        elif element['type']=="CompositeElement" and element['text']!="":
            categorized_elements.append(Element(type="text", text=element['text'], page_label=page_num, file_name=file_name, id=element['id']))
            text_pages.append(page_num)
        
# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

# Page Break
page_elements = [e for e in categorized_elements if e.type == "page_break"]
print(len(page_elements))

# Summary chain
prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
model = ChatOpenAI(temperature=0, model="gpt-4", request_timeout=60)
summarize_chain = {"element": lambda x:x} | prompt | model | StrOutputParser()

find_min_max_length = lambda strings: (len(min(strings, key=lambda x: len(x), default="")), len(max(strings, key=lambda x: len(x), default="")))

# Apply to texts
texts = [i.text for i in text_elements if i.text!=""]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 8})

print("Text Summary Generated")
print(text_summaries[:3])
print(f"length range: {find_min_max_length(text_summaries)}")
assert len(text_summaries) == len(texts)

# Apply to tables
tables = [i.text for i in table_elements if i.text!=""]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 8})

print("Table Summary Generated")
print(table_summaries[:3])
print(f"length range: {find_min_max_length(table_summaries)}")
assert len(table_summaries) == len(tables)

# Load embedding model
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': "cuda:0"}
encode_kwargs = {'normalize_embeddings': True}
embed_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章: "
)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embed_model,
    persist_directory="./chroma_bge",
)
store = RedisStore(redis_url="redis://localhost:6379", client_kwargs={"decode_responses": True})
id_key = "doc_id"
page_label = "page_label"
file_label = "file_name"
ori_text = "ori_text"
summary = "summary"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [e.id for e in text_elements]
summary_texts = [Document(page_content=s, metadata={id_key: doc_ids[i], 
                                                    page_label: text_elements[i].page_label, 
                                                    file_label: text_elements[i].file_name,
                                                    ori_text: texts[i]}) for i, s in enumerate(text_summaries)]
text_docs = [json.dumps({"page_content": s, "metadata": {id_key: doc_ids[i], 
                                                         page_label: text_elements[i].page_label, 
                                                         file_label: text_elements[i].file_name,
                                                         summary: text_summaries[i]}}) for i, s in enumerate(texts)]
retriever.vectorstore.add_documents(summary_texts)
# summary_metadata = [{"page_content": texts, "metadata": t.metadata} for t in summary_texts]
retriever.docstore.mset(list(zip(doc_ids, text_docs)))

# Add tables
table_ids = [e.id for e in table_elements]
summary_tables = [Document(page_content=s, metadata={id_key: table_ids[i], 
                                                     page_label: table_elements[i].page_label, 
                                                     file_label: table_elements[i].file_name,
                                                     ori_text: tables[i]}) for i, s in enumerate(table_summaries)]
table_docs = [json.dumps({"page_content": s, "metadata": {id_key: table_ids[i], 
                                                          page_label: table_elements[i].page_label, 
                                                          file_label: table_elements[i].file_name,
                                                          summary: table_summaries[i]}}) for i, s in enumerate(tables)]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, table_docs)))