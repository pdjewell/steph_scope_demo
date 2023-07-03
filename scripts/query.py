import os
from typing import Any
import streamlit as st
import openai
from langchain.vectorstores import Chroma 
from langchain.vectorstores.chroma import _results_to_docs_and_scores
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


# get openai api key from env
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Get completion helper function 
def get_completion(prompt, model="gpt-3.5-turbo-16k", temperature=0.05):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message["content"]


class QueryVectorstore():

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", 
                                temperature=0.05,
                                openai_api_key=openai.api_key)


    def __call__(self, query, 
                 filter={},
                 guideline_cat=None,
                 k=4, threshold=0.4,
                 max_words=150,
                 temperature=0.05,
                 reformat_results=False,
                 review_consultation=False,):

        results_with_scores = self.get_top_results_with_scores(
                                                query,
                                                filter=filter,
                                                guideline_cat=guideline_cat,
                                                k=k, threshold=threshold,
                                                sort_results=True)

        if len(results_with_scores) == 0:
            response = "There is no information in the available guidelines that can help with this query."
            references = []
            prompt = ""
        else:
            results_with_scores = self.add_references(results_with_scores)
            if review_consultation:
                prompt = self.create_review_consultation_prompt(query, 
                                                results_with_scores,
                                                max_words=max_words,
                                                reformat_results=reformat_results)
            else:
                prompt = self.create_main_prompt(query, 
                                            results_with_scores,
                                            max_words=max_words,
                                            reformat_results=reformat_results)
            
            response = get_completion(prompt, temperature=temperature)
            references = self.group_references(results_with_scores)

        return {'response': response, 
                'references': references,
                'prompt': prompt,
                'results_with_scores': results_with_scores}
    

    def correct_spelling(self, query):

        prompt = f"""
        Below is a query entered by a user, delimited by triple back ticks, and your task is to correct any errors and return the corrected query.
\n
Think step by step and perform the following tasks:
1) Correct any spelling errors
2) Capitalize any proper nouns, names, or acronyms.
3) Expand any acronyms or abbreviations, using the information delimited by triple dashes below if needed, and include the acronym in parentheses after.
\n
---
Some common abbrevations or acronyms to expand:
Rx/rx : treatment
Dx/dx : diagnosis
Abx/abx : antibiotics
CAP/cap : community-acquired pneumonia (CAP)
HAP/hap : hospital-acquired pneumonia (HAP)
AF/af : atrial fibrillation (AF) 
COPD/copd : chronic obstructive pulmonary disease
PC: presenting complaint
HPC : history of presenting complaint
PMH : past medical history (PMH)
DH : drug history (DH)
SH : social history (SH)
NKDA = no known drug allergies (NKDA)
LABA = long-acting beta agonist (LABA)
SABA = short-acting beta agonist (SABA)
ICS = inhaled corticosteroid (ICS)
ICU = intensive care unit (ICU)
ED = emergency department (ED)
SOB = shortness of breath (SOB)
SOBOE = shortness of breath on exertion (SOBOE)
RR = respiratory rate (RR)
HR = heart rate (HR)
BP = blood pressure (BP)
SpO2 = oxygen saturation (SpO2)
O2/o2 = oxygen (O2)
O2 sat = oxygen saturation (O2 sat)
Pt/pt = patient
hx = history
y/o = years old
w/ = with
w/o = without
c/o = complains of
Ca/ca = cancer
---
\n
If there are no corrections, return the query as is.
\n
\n
Some examples to demonstrate:\n
Query: what is the Rx, specifically abx, for sever pnemicoccal neumonia CAP?
Return: What is the treatment, specifically antibiotics, for severe pneumococcal community-acquired pneumonia (CAP)?
\n
Query: how do you Dx AF?
Return: How do you diagnose atrial fibrillation (AF)?
\n
Query: what is the first line LABA Rx for a pt with COPD, who has worsening SOBOE?
Return: What is the first line long-acting beta agonist (LABA) treatment for a patient with chronic obstructive pulmonary disease (COPD), who has worsening shortness of breath on exertion (SOBOE)?
\n
Query: rx for hap?
Return: Treatment for hospital-acquired pneumonia (HAP)?

Here is the user query to correct: 
```{query}```
"""
        response = get_completion(prompt, temperature=0.1)

        return response
    

    def check_query_appropriate(self, query):

        prompt = f"""
        Please check the query below, and determine whether the query is regarding a medical or clinical question.
        If it is not a medical or clinical query, return "NO".
        If it is a medical or clinical query, return "YES".
        If you are unsure, return "YES".
        \n
        For example:
        \n
        Query: what is the treatment, specifically antibiotics, for severe pneumococcal community-acquired pneumonia (CAP)?
        Return: YES
        \n
        Query: is father christmas real?
        Return: NO
        \n
        Query: {query}
        Return:
        """
        response = get_completion(prompt, temperature=0)

        return response


    def results(self, query, filter={},
                 guideline_cat=None,
                 k=3, threshold=0.3,
                 sort_results=False):

        results_with_scores = self.get_top_results_with_scores(
                                                query,
                                                filter=filter,
                                                guideline_cat=guideline_cat,
                                                k=k, threshold=threshold,
                                                sort_results=sort_results)

        if len(results_with_scores) == 0:
            print("No results found")
            return "",""
        else:
            results_with_scores = self.add_references(results_with_scores)
            references = self.group_references(results_with_scores)
            references = "\n".join(references)

            return results_with_scores, references 


    def get_top_results_with_scores(self, 
                                    query, 
                                    filter={},
                                    guideline_cat=None,
                                    k=4, threshold=0.4,
                                    sort_results=False):
    
        if guideline_cat is not None:
            filter={"broad_category": guideline_cat}
        
        results = self.vectorstore._collection.query(
            query_texts=[query],
            n_results=k,
            where=filter)

        # note langchain function used here may change in future
        results_with_scores = _results_to_docs_and_scores(results)

        # Remove results with similarity score above threshold
        score_threshold = threshold 
        results_with_scores = [(doc, similarity)
                                for doc, similarity in results_with_scores
                            if similarity <= score_threshold]
        
        if sort_results:
            results_with_scores = sorted(results_with_scores, 
                                                key=lambda x: (x[0].metadata['file_name'], 
                                                               x[0].metadata['chunk_index']))
        # Remove results if same file_name and page number
        #results_with_scores_unique = []
        #for result, score in results_with_scores:
        #    file_name = result.metadata['file_name']
        #    page = result.metadata['page']
        #    if (file_name, page) not in [(r.metadata['file_name'], r.metadata['page']) for r, s in results_with_scores_unique]:
        #        results_with_scores_unique.append((result, score))

        # Sort results by file_name and chunk index


        print(f"Number of results returned: {len(results_with_scores)}")
        #print(f"Number of unique results returned: {len(results_with_scores_unique)}")

        return results_with_scores
    

    def add_references(self, results_with_scores):

        for result, _ in results_with_scores:
            org = result.metadata['org'].title()
            soc = result.metadata['soc_or_trust'].title()
            cat = result.metadata['broad_category'].title()
            specialty = result.metadata['specialty'].title()
            title = result.metadata['title']
            year = result.metadata['year']
            page = result.metadata['page']
            link = result.metadata['url']

            reference = f"""
            Source: {soc} guidance ({cat} {specialty} guideline), 
            Title: {title}
            Published: {year}
            Page: {page} 
            Link: {link}\n
            """

            result.metadata['reference'] = reference

        return results_with_scores 


    def group_references(self, top_results):

        files = {}
        for result, _ in top_results:
            file_name = result.metadata['file_name']
            page = int(result.metadata['page']) + 1 # plus 1 as pdf page numbers start at 1

            if file_name not in files:
                org = result.metadata['org'].upper()
                soc = result.metadata['soc_or_trust'].title()
                cat = result.metadata['broad_category'].title()
                specialty = result.metadata['specialty'].title()
                title = result.metadata['title']
                year = result.metadata['year']
                link = result.metadata['url']

                files[file_name] = {
                    "org": org,
                    "soc": soc,
                    "cat": cat,
                    "specialty": specialty,
                    "title": title,
                    "year": year,
                    "link": link,
                    "pages": set([page])
                }
            else:
                files[file_name]["pages"].add(page)

        refs = []
        for file in files.values():
            pages = sorted(file["pages"])
            reference = f"""
            Source: {file['soc']} guidance ({file['cat']} {file['specialty']} guideline), 
            Title: {file['title']}
            Published: {file['year']}
            Page(s): {', '.join(str(p) for p in pages)} 
            Link: {file['link']}\n
            """
            refs.append(reference)

        return refs
    
    

    def get_chunk_before(self, result):

        file_name = result.metadata['file_name']
        chunk_index = result.metadata['chunk_index']

        if chunk_index != 0: 

            chunk_before_index = str(int(chunk_index) - 1) 

            filter = {"$and": [
                {"file_name": {"$eq": file_name}},
                {"chunk_index": {"$eq": chunk_before_index}}]}

            chunk_before = self.vectorstore._collection.get(where=filter)['documents'][0]
            
            print("Successfully returned chunk before current chunk")
            return chunk_before

        else:
            print("No chunk before returned, as current chunk is the first of document")
            return "" 
    

    def get_chunk_after(self, result):

        file_name = result.metadata['file_name']
        chunk_index = result.metadata['chunk_index']
        last_chunk_index = result.metadata['last_chunk_index']

        if chunk_index != last_chunk_index:

            chunk_after_index = str(int(chunk_index) + 1) 

            filter = {"$and": [
                {"file_name": {"$eq": file_name}},
                {"chunk_index": {"$eq": chunk_after_index}}]}

            chunk_after = self.vectorstore._collection.get(where=filter)['documents'][0]

            print("Successfully returned chunk after current chunk")
            return chunk_after
        
        else:
            print("No chunk after returned, as current chunk is last of document")
            return ""
        

    def combine_all_chunks(self, results_with_scores):

        # sort results 
        results_with_scores = sorted(results_with_scores, 
                                    key=lambda x: (x[0].metadata['file_name'], 
                                            x[0].metadata['chunk_index']))
        # Get chunk indexes from results
        chunk_indices = [(r.metadata['file_name'], r.metadata['chunk_index']) for r, _ in results_with_scores]
        
        chunks_combined = []
        chunk_indices_added = []
        pages_added = []
        for result, _ in results_with_scores:
            # set file_name and chunk indices before after 
            file_name = result.metadata['file_name']
            page = result.metadata['page']
            chunk_index_before = str(int(result.metadata['chunk_index']) - 1)
            chunk_index_current = result.metadata['chunk_index']
            chunk_index_after = str(int(result.metadata['chunk_index']) + 1)
            # check if chunk before in list, if not add
            if f"{file_name}_{chunk_index_before}" not in chunk_indices_added:
                chunk_before = self.get_chunk_before(result)
                chunk_indices_added.append(f"{file_name}_{chunk_index_before}")
            else:
                chunk_before = ""
            # check if current chunk in list, if not add
            if (f"{file_name}_{chunk_index_current}") not in chunk_indices_added:
                chunk_current = result.page_content
                chunk_indices_added.append(f"{file_name}_{chunk_index_current}")
            else:
                chunk_current = ""
            # check if chunk after in list, if not add
            if f"{file_name}_{chunk_index_after}" not in chunk_indices_added:
                chunk_after = self.get_chunk_after(result)
                chunk_indices_added.append(f"{file_name}_{chunk_index_after}")
            else:
                chunk_after = ""

            # get metadata
            page = result.metadata['page']
            title = result.metadata['title']
            soc = result.metadata['soc_or_trust'].title()
            cat = result.metadata['broad_category']
            specialty = result.metadata['specialty']
            year = result.metadata['year']
            # write header
            if (file_name, page) not in pages_added:
                header = f"Page {page} of {soc} {cat} {specialty} guideline, entitled {title}, published {year}"
                pages_added.append((file_name, page))
            else:
                header = ""
            # combine everything 
            combined = "\n".join([header, chunk_before, chunk_current, chunk_after])
            chunks_combined.append(combined)

        print(f"chunk_indices_added: {chunk_indices_added}")
        print(f"number of chunks: {len(chunk_indices_added)}")
        print(f"pages_added: {pages_added}")
        chunks_combined = "\n\n".join(chunks_combined)

        return chunks_combined

    
    # OLDER FUNCTION - will delete if new one works 
    def combine_chunks_and_ref(self, result):

        chunk_before = self.get_chunk_before(result)
        chunk_after = self.get_chunk_after(result)

        page = result.metadata['page']
        title = result.metadata['title']
        soc = result.metadata['soc_or_trust'].title()
        cat = result.metadata['broad_category']
        specialty = result.metadata['specialty']
        year = result.metadata['year']

        header = f"Page {page} of {soc} {cat} {specialty} guideline, entitled {title}, published {year}"

        combined_content = "\n" + chunk_before + '\n' + result.page_content + '\n' + chunk_after
        content_with_ref = '\n' + header + '\n' + combined_content

        return content_with_ref


    def create_reformat_results_prompt(self, combined_results: str) -> str:

        reformat_results_prompt =f"""
        You are provided with different sections from clinical guideline documents below, delimited by triple backticks (```).
        They are taken from a search and there may be problems with formatting and overlap.
        \n\n
        Your tasks are as follows in this order:
        1) Identify section headings and make sure they are on a new line and are clearly marked as headings.
        2) Correct any formatting issues, such as incorrect line breaks or spacing.
        3) Remove any overlap or combine sections if required.
        4) Do add information or take information away unless it is overlapping.
        \n\n
        Here is the content to be reformatted:
        ```
        {combined_results}
        ```
        """

        return reformat_results_prompt



    def create_main_prompt(self, 
                      query, 
                      results_with_scores,
                      max_words=100,
                      reformat_results=False,):

        # combine the top result chunks with above and below chunks, and reference
        combined_results = self.combine_all_chunks(results_with_scores)

        # Reformat the combined results for the main prompt
        if reformat_results:
            reformat_results_prompt = self.create_reformat_results_prompt(combined_results)
            combined_results = get_completion(reformat_results_prompt)
        else:
            combined_results = combined_results

        guideline_cat = results_with_scores[0][0].metadata['broad_category'].title()

        prompt = f"""
        You are an intelligent medical doctor, helping another doctor with the following clinical query, delimited by triple back ticks:
        ```
        {query}
        ```

        Provide a concise clear bullet-pointed professional response directly answering the query. \n
        Using relevant information provided by the provided clinical guidelines sections below, delimited by triple dashes. \n
        Use headings in the response to clearly mark different sections if required. \n
        Think step by step. \n 
        Note as these are sections of guidelines, some information may be cut off. \n
        Use a maximum of {max_words} words in your response. \n
        If no information is provided within the triple dashes, simply respond with:
        'I am unable to answer the query with information from the available {guideline_cat} guidelines'.
        \n
        Here is some information from {guideline_cat} clinical guidelines to answer the query, delimited by triple dashes: 
        ---
        {combined_results}
        ---
        \n
        The other doctor's query is again as follows, delimited by triple back ticks: 
        ```
        {query}
        ```
        Use a maximum of {max_words} words in your response.
        """

        return prompt
    

    def create_review_consultation_prompt(self,
                                    consult_summary, 
                                        results_with_scores,
                                        max_words=200,):

        # combine the top result chunks with above and below chunks, and reference
        combined_results = self.combine_all_chunks(results_with_scores)

        prompt = f"""
        You are an intelligent medical doctor. Another doctor has just had a consultation with patient. A brief summary of this consultation, including their diagnosis / impression and management plan, is provided below, delimited by triple back ticks:\n
        ```
        {consult_summary}
        ```

        Your task is to use the information from the clinical guidelines below, delimited by triple dashes, to provide a response to the other doctor, regarding the diagnosis / impression and management plan. \n
        If their are inconsistencies between the diagnosis / impression and management plan, with the guidelines, you should highlight these. \n
        Make suggestions or corrections to the diagnosis and/or management plan, if required. \n 
        Check drug doses for example. \n
        Think step by step. \n
        Give a concise clear bullet-pointed professional response. \n
        Use a maximum of {max_words} words in your response. \n

        If no information is provided within the triple dashes, simply respond with:
        'I am unable to answer the query with information from the available guidelines'.
        \n
        Here is the information from the clinical guidelines to use to critique the diagnosis and management plan, delimited by triple dashes:\n
        ---
        {combined_results}
        ---
        \n
        """

        return prompt