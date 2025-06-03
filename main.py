import streamlit as st
#from dependencies import *
from textranksummariser import textrank_summarizer_nltk
from tfidf import tfidf_summarizer
# from pegasus import pegasus_summary
from transformers import PegasusTokenizer, pipeline
# model_name = "google/pegasus-xsum"
# pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
# summarizer = pipeline(
#                     "summarization",
#                     model=model_name,
#                     tokenizer=pegasus_tokenizer, 
#                     framework="pt"
#                     )
# tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
# model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

def get_summary(text, model):
    match model:
        case "textrank-summariser":
            summary = textrank_summarizer_nltk(text)
            st.session_state.summary =  summary
        case "TFIDF-summariser":
            summary = tfidf_summarizer(text)
            st.session_state.summary =  summary
        case "pegasus-summariser":
            with st.spinner('Preparing summary. Please wait...'):
                model_name = "google/pegasus-xsum"
                pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
                summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=pegasus_tokenizer, 
                    framework="pt"
                    )

                summary = summarizer(text, min_length=30, max_length=150)[0]["summary_text"]
                # summary = pegasus_summary(text)

                st.session_state.summary =  summary

    # st.subheader("Summary")
    # st.write(st.session_state.summary)



if "summary" not in st.session_state:
    st.session_state.summary = ""
# if "text"not in st.session_state:
#     st.session_state.text = ""

st.set_page_config("Summariser" , page_icon=":shark:")

st.header("Summariser")

text = st.text_area("Enter paragraph to summarise" , height = 250)

model = st.selectbox("Summariser" , ["textrank-summariser" , "TFIDF-summariser" , "pegasus-summariser"])

submit = st.button(label ="Summarise", on_click=get_summary , args=(text, model))

if submit:
    st.subheader("Summary")
    st.write(st.session_state.summary)


    
    # st.subheader("Summary")
    # st.write(st.session_state.summary)
    


    # if submit:
    #     if model == "textrank-summariser":
    #         summary = textrank_summarizer_nltk(text)
    #         st.subheader("Summary")
    #         st.write(summary)
    #     elif model == "TFIDF-summariser":
    #         summary = tfidf_summarizer(text)
    #         st.subheader("Summary")
    #         st.write(summary)
    #     elif model == "pegasus-summariser":
    #         with st.spinner('Preparing summary. Please wait...'):
    #             model_name = "google/pegasus-xsum"
    #             pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
    #             pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
    #             summarizer = pipeline(
    #                 "summarization",
    #                 model=model_name,
    #                 tokenizer=pegasus_tokenizer, 
    #                 framework="pt"
    #                 )
    #             summary = summarizer(text, min_length=30, max_length=150)

    #             st.subheader("Summary")
    #             st.write(summary[0]["summary_text"])

    