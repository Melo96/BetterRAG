SUMMARY_PROMPT = """
    You are an assistant tasked with summarizing tables and text. 
    Give a concise summary of the table or text in Chinese. 
    Table or text chunk: {element}
"""

# SUMMARY_PROMPT = """
#     Generate a concise and coherent summary from the given Context. 
#     Condense the context into a well-written summary that captures the main ideas, key points, and insights presented in the context. 
#     Prioritize clarity and brevity while retaining the essential information. 
#     Aim to convey the context's core message and any supporting details that contribute to a comprehensive understanding. 
#     Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the context. 
#     Provide context where necessary and avoid excessive technical jargon or verbosity.
#     The goal is to create a summary that effectively communicates the context's content while being easily digestible and engaging.

#     CONTEXT: {context}

#     SUMMARY: 
# """

MULTI_QUERY_PROMPT = """
    You are an AI language model assistant.
    Your task is to generate {queryCount} different versions of the given user
    question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search.
    The generated questions should be in Chinese.

    Provide these alternative questions separated by newlines between XML tags. For example:

    <questions>
    Question 1
    Question 2
    Question 3
    </questions>

    Original question: {question}
"""

QUERY_PLANNER_PROMPT = """
    You are a world class query planning algorithm capable of breaking apart questions into its dependency queries 
    such that the answers can be used to inform the parent question.
    Do not answer the questions, simply provide a correct compute graph with good specific questions to ask and relevant dependencies. 
    Your need to generate {queryCount} of these dependency queries.
    Before you call the function, think step-by-step to get a better understanding of the problem.
    The generated queries should be in Chinese.

    Provide these queries separated by newlines between XML tags. For example:

    <questions>
    Question 1
    Question 2
    Question 3
    </questions>

    Original question: {question}
    Dependency queries:
"""

RAG_PROMPT = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    Your answer should be as comprehensive as possible.
    Think step-by-step before you answer the question. 
    If you don't know the answer, just say that you don't know. 
    The returned answer should be in Chinese. 
    Question: {question} 
    Context: {context} 
    Answer:
"""

FILTER_PROMPT = """
    You are an assistant for extracting keywords from query.
    All possible keywords are listed below:
    keywords: DC宽温系列, DM多轴系列, DN单板系列, DP精巧系统, DR环形系列, Dservo系列, DS常规系列, UF2 Uservo-Flex系列双编版本, Uservo-Flex系列
    
    Your task is to extract one or multiple keywords from the query.
    Return the extracted keywords in the following format:
    keyword1|keyword2|keyword3
    If you cannot extract any keywords from query, please return all of the possible keywords. 
    
    Here are a few examples:
    Query: DC宽温系列的产品有哪些？
    Resposne: DC宽温系列
    
    Query: Uservo-Flex系列EtherCAT驱动器的模拟量输入接线方法？
    Resposne: Uservo-Flex系列|UF2 Uservo-Flex系列双编版本
    
    Query: 磁环怎么接?
    Resposne: DC宽温系列|DM多轴系列|DN单板系列|DP精巧系统|DR环形系列|Dservo系列|DS常规系列|UF2 Uservo-Flex系列双编版本|Uservo-Flex系列
    
    Task begin!
    Query: {query}
    Response:
"""