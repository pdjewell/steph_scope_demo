import os
import streamlit as st
import openai
from langchain.vectorstores import Chroma 
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from scripts.loadvectorstore import VectorStoreLoader
from scripts.query import QueryVectorstore
import base64
import fitz
from PIL import Image

# streamlit app config and style 
@st.cache_data
def load_css(file_name = "./style/style.css"):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css

icon = Image.open('./images/small_app_icon.png')
st.set_page_config(page_title="StephScope", 
                   page_icon=icon,
                   layout="wide")
css = load_css()
st.markdown(css, unsafe_allow_html=True)

logo = Image.open("images/scopelogo4.png")
# Resize the image to a smaller size
width, height = logo.size
resized_image = logo.resize((int(width * 0.5), int(height * 0.5)))
st.header('')
st.image(resized_image, caption='', width=500)

#title = '🔎StephScope'
#st.markdown(f"<h1 style='text-align: center; color: #006666;'>{title}</h1>", unsafe_allow_html=True)


# load and cache vectorstore
@st.cache_resource
def load_vectorstore(vector_db_directory=os.path.join(os.getcwd(), 'vector_db'),
                     vectorstore_name=
                     'chroma_db_ada_embeddings'):
    
    vectorstore_loader = VectorStoreLoader()
    vectorstores = vectorstore_loader.load_all_chroma_db()
    vectorstore = vectorstores[vectorstore_name]

    return vectorstore


# display PDF in app at correct page
def displayPDF(file_path, page):
    # Opening file from file path
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # adjust width to be half of screen width
    width, height = 700, 900
    # adjust page to + 1 as pdf starts page at 1
    page = page + 1
    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# display image of PDF page
def show_page(file_path, page):
    # Open the PDF document using fitz
    doc = fitz.open(os.path.join(file_path))   
    # Get the specific page to render
    page = doc[page]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    # Create an Image object from the rendered pixel data
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image 


def main():
    pass


if __name__ == "__main__":

    vector_db_directory = os.path.join(os.getcwd(), 'vector_db')
    vectorstore = load_vectorstore(vector_db_directory=vector_db_directory,
                                   vectorstore_name="chroma_db_ada_embeddings")

    # User input for query
    query = st.text_input(label="Enter your query",
                          value="Enter your query here",
                          label_visibility="collapsed")     
        
    if st.button('Submit'):

        query_vs = QueryVectorstore(vectorstore) 

        # Correct query spelling
        query = query_vs.correct_spelling(query)

        # Check query is appropriate
        with st.spinner('Checking query...'):
            appropriate = query_vs.check_query_appropriate(query)

            if appropriate.strip().upper() == 'NO':
                st.write('Please enter an appropriate clinical query')
                st.stop()

            # If deemend appropriate, continue
            else:
                st.write(f"Query: {query}")
                  
        # create tabs
        tabs = ["Local guidance", "National and International guidelines", "PubMed"]
        tab1, tab2, tab3 = st.tabs(tabs)

        ## LOCAL GUIDANCE TAB
        with tab1:

            with st.spinner('Searching local guidelines...'):
                # Query vectorstore: local guidance
                results_with_scores, references = query_vs.results(query,
                                                                filter={"broad_category": 'local'},
                                                                    k=3, threshold=0.3,
                                                                    sort_results=False)
                # If results found, continue
                if bool(results_with_scores):

                    # set file_path and page of top result:
                    file_path = results_with_scores[0][0].metadata['source']
                    page = results_with_scores[0][0].metadata['page']

                    # create columns
                    col1, col2 = st.columns(2)

                    # PDF VIEWER
                    with col2:
                        # Display pdf of top result
                        st.write('Top results:')
                        st.write(references)
                        displayPDF(file_path, page)

                    # TEXT VIEWER
                    with col1:

                        # Query local guidance 
                        response = query_vs(query,
                                                filter={"broad_category": 'local'},
                                                k=3, threshold=0.3)
                        st.write(response['response'])
                        st.write('\nReferences:')
                        refs = response['references']
                        for ref in refs:
                            st.write(ref)
                        # # For debugging
                        #st.write(f'\nPrompt length: {len(response["prompt"])})')
                        #st.write('\nPrompt:')
                        #st.text(response['prompt'])

                # If no results found, display message
                else:
                    st.write('No results found in local guidance for this query')

            
        
        ## NATIONAL/INTERNATIONAL GUIDANCE TAB
        with tab2: 
            with st.spinner('Searching national and international guidelines...'): 

                # Query vectorstore: national and international guidance
                filter = {"$or": [
                            {"broad_category": {"$eq": "national"}},
                            {"broad_category": {"$eq": "international"}}]}
                results_with_scores, references = query_vs.results(query,
                                                                filter=filter,
                                                                    k=3, threshold=0.3,
                                                                    sort_results=False)
                
                # If results found, continue, else display message
                if bool(results_with_scores):

                    # set file_path and page of top result:
                    file_path = results_with_scores[0][0].metadata['source']
                    page = results_with_scores[0][0].metadata['page']
                
                    # create columns
                    col3, col4 = st.columns(2)

                    # PDF VIEWER
                    with col4:
                        # Display pdf of top result
                        st.write('Top results:')
                        st.write(references)
                        displayPDF(file_path, page)

                    # TEXT VIEWER
                    with col3:
                        # Query national and international guidance 
                        filter = {"$or": [
                            {"broad_category": {"$eq": "national"}},
                            {"broad_category": {"$eq": "international"}}]}
                        response = query_vs(query,
                                                filter=filter,
                                                k=3, threshold=0.3)
                        st.write(response['response'])
                        st.write('\nReferences:')
                        refs = response['references']
                        for ref in refs:
                            st.write(ref)  
                        # For debugging
                        #st.write(f'\nPrompt length: {len(response["prompt"])})')
                        #st.write('\nPrompt:')
                        #st.text(response['prompt'])

                # If no results found, display message
                else:
                    st.write('No results found in national/international guidance for this query')

        ## PUBMED TAB
        with tab3: 
            with st.spinner('Searching PubMed...'):
                 
                # Query pubmed
                from langchain.retrievers import PubMedRetriever
                from langchain.utilities import PubMedAPIWrapper
                retriever = PubMedRetriever()
                response = retriever.get_relevant_documents(query)

                if response:
                    number_of_results = len(response)
                    max_number_to_display = 1
                    number_to_display = min(number_of_results, max_number_to_display)
                    st.write("Displaying top results from PubMed:")
                    for i in range(number_to_display):
                        st.write(response[i].metadata['title'])
                        uid = response[i].metadata['uid']
                        st.write(f"https://pubmed.ncbi.nlm.nih.gov/{uid}/")
                        st.write("Abstract:")
                        st.write(response[i].page_content)
                else:
                    st.write('No results found in PubMed for this query')

