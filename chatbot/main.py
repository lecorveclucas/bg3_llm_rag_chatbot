# Ideas - TO-DO list
# Build a SIMPLE model/RAG
# Containerize it
# Make a proper README.md
# Build a scrapper to scrap data from BG3
# Build a web app (through chatgpt4)
# Create agent depending on which db to use ? character, weapons, skills, etc. ?

import os
import streamlit as st
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 10},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True
)

def get_build_index(documents, llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3,
                    save_dir="./vector_store/index"):
    node_parser = SentenceWindowNodeParser(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    # sentence_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    #     node_parser=node_parser,
    # )
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        # create and load the index
        index = VectorStoreIndex.from_documents([documents])
        index.storage_context.persist(persist_dir=save_dir)
    else:
        # load the existing index
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir))

    return index

def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base", device="mps"
    )
    engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )

    return engine



st.header("Chat with the Baldur's Gate 3 wiki 🎲 🎮")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Baldur's Gate 3"}
    ]

@st.cache_resource(show_spinner=False)
def load_db():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        documents = SimpleDirectoryReader(input_files=["./data/2303.18223.pdf"]).load_data()
        documents = Document(text="\n\n".join([doc.text for doc in documents]))

        # Get the vector index
        vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5",
                                        sentence_window_size=3, save_dir="./vector_store/index")
        return vector_index
    

vector_index = load_db()

query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=6, rerank_top_n=2)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(st.session_state.messages[-1]["content"])
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history





