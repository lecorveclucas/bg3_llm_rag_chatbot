import os
import streamlit as st
import glob
import base64
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


files = glob.glob("./data/*/*.txt")
documents = SimpleDirectoryReader(input_files=files).load_data()

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf',
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q6_K.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=512,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 15},
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

vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5",
                                sentence_window_size=3, save_dir="data/vector_store/index")

import json
with open("./data/vector_store/index/default__vector_store.json", "r") as f:
    test = json.load(f)

print(len(test["embedding_dict"].keys()))

# print(vector_index.docstore.docs["6e42b7c4-a549-46b6-8e57-a971e1909615"].text)

# print(vector_index.docstore.docs["adb9f63c-777a-40e6-bebf-605219b603e0"].text)

# print(vector_index.docstore.docs["1d127679-bb1e-4c85-9fa1-375633a0be7d"].text)

# print(vector_index.docstore.docs["087b0201-896a-409f-93a1-3280dfd51422"].text)

# print(vector_index.docstore.docs["6eb52c32-9e80-4221-a124-33d01ed890f7"].text)

# print(vector_index.docstore.docs["bdb73309-20b9-4560-b339-8399a35a960a"].text)

# print(vector_index.docstore.docs["e416d3a9-7c01-4bbb-a6c3-1dfb98b619e3"].text)

# print(vector_index.docstore.docs["e0e32ba6-a152-4d51-a975-a35a77b3636d"].text)

# print(vector_index.docstore.docs["b3592878-dadc-48ad-9e86-5e784ea240b9"].text)

print(vector_index.docstore.docs["bc5f03d8-8ffb-461b-8fd5-a84e396cccb8"].text)
