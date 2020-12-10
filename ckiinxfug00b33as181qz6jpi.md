## Named Entity Recognition App using Streamlit, Spacy and Python

This blog article highlights the implementation of Named Entity Recognition with SpaCy. 

Below is the snapshot of how our app looks like:
![ner.gif](https://cdn.hashnode.com/res/hashnode/image/upload/v1604001297209/9QaLFYSHo.gif)

### Table of Contents
1. Introduction to Natural Language Processing and Named Entity Recognition
2. Tools and Libraries
3. Pre-requisites
4. Structure and workflow of the app
5. Code (Python Implementation)
6. Testing

### Introduction to Natural Language Processing and Named Entity Recognition
#### Natural Language Processing-
Natural language is the language used by humans to communicate with each other (verbally or textually - for instance, face-to-face conversations, blogs, emails, SMS messages, etc.). Natural language is incredibly important for computers to understand for some reasons as under:

- It can be viewed as a source of huge amount of data, and if processed intelligently by the system, can yield the useful information.
- It can be helpful for computers to better communicate with humans.

However, unlike humans, computers can't easily comprehend natural language. Sophisticated methods and techniques are needed to translate natural language into a format understandable by computers. ***Natural Language Processing*** (NLP) helps us achieve this objective. 

NLP covers the following concepts:
- Lemmatization
- Stemming
- Parts-Of-Speech Tagging
- Named Entity Recognition
- Bag of Words Approach
- TF-IDF
- N Grams

In this app, we'll dive into the concept of *Named Entity Recognition*.

#### Named Entity Recognition
*Named Entity Recognition* is the process of classifying entities into predefined categories such as person, date, time, location, organization, percentage etc.

*Example*- In the sentence "Mark Elliot Zuckerberg (born May 14, 1984) is known for co-founding Facebook Inc. . He is having his talk in America at 11 a.m.". The Named Entity Recognizer will give following information about this sentence:

>Mark Elliot Zuckerberg -> PERSON <br>
May 14, 1984 -> DATE <br>
Facebook Inc. -> Organization (ORG) <br>
America -> Location (GPE) <br>
11 a.m. -> TIME

### Tools and Libraries
#### 1. SpaCy-
It is one of the most popular NLP libraries. spaCy was developed in 2015. spaCy is known to be the fastest NLP framework in Python, with single optimised functions for each of the NLP tasks it implements. Being easy to learn and use, one can easily perform simple tasks using a few lines of code.

It provides a default model which can recognise a wide range of named entities, including person, organization, percentage, language, event etc. Apart from these default entities, we have the liberty to add arbitrary classes to the NER model, after training the model to update it with newer trained examples.

#### 2. Streamlit-
Streamlit is an open-source Python library that has been used in our app for the user interface. 
1. Make sure you have Python 3.6 or greater installed.
2. Install Streamlit using the command (in terminal):
```
pip install streamlit
```

#### 3. SpaCy_Streamlit-
Spacy-Streamlit is a python package used for visualizing spaCy models and to build interactive spaCy-powered apps with Streamlit.

It has numerous functions for visualizing spacy’s essential NLP features such as-

- NER(Named Entity Recognition) using `visualize_ner()`
- Tokenization using `visualize_tokens()`
- Parser using `visualize_parser()`
- Text Categorizer using `visualize_textcat()`
- Sentence Similarity using `visualize_similarity()`
and many more...

In our app, we have focused on ***`spacy_streamlit.visualize_ner()`*** to visualize the Named Entities using spaCy model.


### Pre-requisites
- Download and Install Python 3.6 or greater.
- Install Streamlit from the terminal using the command-
```
pip install streamlit 
``` 
- Install spacy_streamlit and spacy using the commands (on terminal)-
```
pip install spacy_streamlit spacy
python -m spacy download en
```

### Structure and workflow of the app
The app highlights two main functions- 
1. Named Entity Recognition from text.
2. Named Entity Recognition from URL.

We can identify named entities from the text provided OR we can provide a URL from which the app fetches the text and extract the named entities.

### The code
Let's dive straight into the code-
```
#Core Pkgs
import streamlit as st

#NLP Pkgs
import spacy_streamlit
import spacy
nlp = spacy.load('en')

#Web Scraping Pkgs
from bs4 import BeautifulSoup
from urllib.request import urlopen

@st.cache
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = " ".join(map(lambda p:p.text, soup.find_all('p')))
	return fetched_text


def main():
	"""A Simple NLP App with Spacy-Streamlit"""
	st.title("Named Entity Recognition")
	
	menu = ["NER", "NER for URL"]
	choice = st.sidebar.radio("Pick a choice", menu)


	if choice == "NER":
		raw_text = st.text_area("Enter Text","")
		if raw_text != "":
			docx = nlp(raw_text)
			spacy_streamlit.visualize_ner(docx, labels = nlp.get_pipe('ner').labels)

	elif choice == "NER for URL":
		raw_url = st.text_input("Enter URL","")
		text_length = st.slider("Length to Preview", 50,200)
		if raw_url != "":
			result = get_text(raw_url)
			len_of_full_text = len(result)
			len_of_short_text = round(len(result)/text_length)
			st.subheader("Text to be analyzed:")
			st.write(result[:len_of_short_text])
			preview_docx = nlp(result[:len_of_short_text])
			spacy_streamlit.visualize_ner(preview_docx, labels = nlp.get_pipe('ner').labels)

if __name__ == '__main__':
	main()
```
Breaking it down,
```
#Core Pkgs
import streamlit as st

#NLP Pkgs
import spacy_streamlit
import spacy
nlp = spacy.load('en')

#Web Scraping Pkgs
from bs4 import BeautifulSoup
from urllib.request import urlopen
```
The first step is to import various packages, libraries and frameworks to be used in our code. Web Scraping Packages are used to fetch the text from URL provided. NLP Packages are used to extract the named entities from the text. 

```
@st.cache
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = " ".join(map(lambda p:p.text, soup.find_all('p')))
	return fetched_text
```
Here, we have defined a function `get_text()` and provided `raw_url` as its parameter. Using this function, we extract or fetch text from a given URL using **BeautifulSoup** and **urllib.request** .

```
def main():
	"""A Simple NLP App with Spacy-Streamlit"""
	st.title("Named Entity Recognition")
	
	menu = ["NER", "NER for URL"]
	choice = st.sidebar.radio("Pick a choice", menu)
```
Next, we define our `main()` function and then provide the title to our app using `st.title("Named Entity Recognition")`. Then, on the sidebar, we have asked for a choice from the user between *Extracting Named Entities from a Text Input* OR *Extracting Named Entities from the Text fetched from URL provided*.

```
	if choice == "NER":
		raw_text = st.text_area("Enter Text","")
		if raw_text != "":
			docx = nlp(raw_text)
			spacy_streamlit.visualize_ner(docx, labels = nlp.get_pipe('ner').labels)
```
So, if user makes a choice as `NER`, he'll be asked to *Enter Text* using `st.text_area("Enter Text",)`. We can also provide a default text (say, "Srishti Gupta (born 24 September 1999) works as a Python developer") using `st.text_area("Enter Text","Srishti Gupta (born 24 September 1999) works as a Python developer")`.<br><br>
In our code, we haven't provided any default text. So if user writes something and presses *Ctrl+Enter*, then we create a nlp object `docx = nlp(raw_text)` which was initialised earlier `nlp = spacy.load('en')`. <br><br>
And then the app visualises any named entity in the text using the `space_streamlit.visualize_ner( )` function. With this function, we can even omit entity labels that we don’t want to be recognized.

```
	elif choice == "NER for URL":
		raw_url = st.text_input("Enter URL","")
		text_length = st.slider("Length to Preview", 50,200)
		if raw_url != "":
			result = get_text(raw_url)
			len_of_full_text = len(result)
			len_of_short_text = round(len(result)/text_length)
			st.subheader("Text to be analyzed:")
			st.write(result[:len_of_short_text])
			preview_docx = nlp(result[:len_of_short_text])
			spacy_streamlit.visualize_ner(preview_docx, labels = nlp.get_pipe('ner').labels)

if __name__ == '__main__':
	main()
```

Else if the choice made by user is `NER for URL`, then the user is prompted to type in the URL. Next, we have used the slider to select the the text length. Then the text is fetched from the URL and then the app displays the **Text to be analyzed** depending upon the `text_length`. <br><br>
Then we create a nlp object `preview_docx = nlp(result[:len_of_short_text)` and then the app visualises any named entity in the text using the `space_streamlit.visualize_ner( )` function. With this function, we can even omit entity labels that we don’t want to be recognised.<br><br>
At last, we call our `main()` function.

### Test the app
To test the app, save the above python code with the name, say, `app.py`. Then, in the terminal, write-

```
streamlit run app.py
```

Thankyou for reading, I would love to connect with you at  [LinkedIn](https://www.linkedin.com/in/srishtii24/). <br><br>