import os
import streamlit as st
import glob
import base64
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank


def get_build_index(documents, llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=5,
                    save_dir="./vector_store/index"):
    node_parser = SentenceWindowNodeParser(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

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

vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5",
                                sentence_window_size=3, save_dir="data/vector_store/index")

import json
with open("./data/vector_store/index/default__vector_store.json", "r") as f:
    test = json.load(f)

print(len(test["embedding_dict"].keys()))
