import streamlit as st
from sentence_transformers import SentenceTransformer

from papertrail import config


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)
