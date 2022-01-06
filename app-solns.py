#imports
from transformers import pipeline
import streamlit as st
import pandas as pd
import os

#setup pipeline
txt_class = pipeline('text-classification', model='charreaubell/distilbert-magazine-classifier', use_auth_token = os.environ["hf_token"])
user_text = st.text_input('Add the text for users to upload!')

#if the text field has something in it, inference and show the output
if user_text:
    res = txt_class(user_text)
    st.write(pd.DataFrame(res, index=[0]))