# Ideas - TO-DO list
# OK : Build a SIMPLE model/RAG
# Containerize it
# Make a proper README.md
# OK : Build a scrapper to scrap data from BG3
# OK : Build a web app (through chatgpt4)
# Create agent depending on which db to use ? character, weapons, skills, etc. ?
# Search to determine how to evaluate this kind of rag ==> Ask 10 questions and compare answers to baseline (ROUGE ? or even with CHATGPT ?) TIME answering time !
# OK : Plot a graph if tokens distribution in sentences  ==> 90% of paragraphs have less than 530 tokens and less than 390 words

# RAG tuning 
# 1. Too much documents from retriever seems to make it hallucinate
# 2. The pre prompt help really much to avoid hallucinating and too be factual ==> Preprompt has a negative impacts on model's reponse as it embeds the preprompt and so it impacts the semantic search in the RAG
# 3. The history might distrub the model answer, to investigate
# 4. Try to divide by chunks, embed only the question THEN give the context to a prepromt
# 5. Names are not embedded, resulting in confusion in match hit with the RAG
# 6. Try hybrid search (with keywords retriever)
# 7. The SemanticSplitterNodeParser seems to increase models accuracy, but the database construction and response time is higher ==> Confirm it with a proper evaluation  ==> Yes it does
# 8. Evaluate model's answers through chatGPT : try to change model's, preprompt, etc.

import os
import streamlit as st
import glob
import base64
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

text_color = "white"

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./assets/bg3_background2.jpg')

# Load custom CSS file
with open("assets/styles.css") as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf',
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',  # Q6_K was used too but quite slow
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.0,  # Model needs to be factual and deterministic
    max_new_tokens=512,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 10},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True
)

def get_build_index(documents, llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=5,
                    save_dir="./vector_store/index"):
    sentence_window_parser = SentenceWindowNodeParser(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text")

    embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=128,
    normalize=True)

    semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1, 
    breakpoint_percentile_threshold=95, 
    embed_model=embed_model)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = semantic_splitter

    if not os.path.exists(save_dir):
        # create and load the index
        index = VectorStoreIndex.from_documents([documents], show_progress=True)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        # load the existing index
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir))

    return index

def get_query_engine(sentence_index, similarity_top_k=3, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base", device="mps"
    )
    engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )

    return engine

def queries_memory(session_state: st.session_state, pre_prompt: str):
    messages_history = session_state.messages
    if len(messages_history) > 2:
        pre_prompt += f"User: {messages_history[-3]["content"]}\n"
        pre_prompt += f"Assistant: {messages_history[-2]["content"]}\n"
    pre_prompt += f"Question: {st.session_state.messages[-1]["content"]}"
    return pre_prompt


# st.markdown('<style>.chat-input { background-color: grey; }</style>', unsafe_allow_html=True)

st.header("Chat with the Baldur's Gate 3 wiki 🎲 🎮")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Baldur's Gate 3"}
    ]

@st.cache_resource(show_spinner=False)
def load_db():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        files = glob.glob("./data/*/*.txt")
        documents = SimpleDirectoryReader(input_files=files).load_data()
        documents = Document(text="\n\n".join([doc.text for doc in documents]))

        # Get the vector index
        vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5",
                                        sentence_window_size=3, save_dir="data/vector_store/index")
        return vector_index
    

vector_index = load_db()

query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=3, rerank_top_n=2)

# Preprompt
pre_prompt = """<s>[INST] Your job is to use information about the game 
Baldur's Gate 3 (BG3) in order to answer questions. Use the given
context to answer questions. Be as detailed as possible, but 
don't make up any information that's not from the context. 
If you don't know an answer, say you don't know.</s>

[INST]
Question : {question}
Context : 
[/INST]
"""

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(f":{text_color}[{message['content']}]")

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # context = queries_memory(session_state=st.session_state, pre_prompt=pre_prompt)
            question = st.session_state.messages[-1]["content"]
            # response = query_engine.query(pre_prompt.format(question=question))
            response = query_engine.query(question)
            st.write(f":{text_color}[{response.response}]")
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

