## Parts-of-Speech Tagging app using Streamlit, spaCy and Python

This blog article highlights the implementation of Part-of-Speech Tagging with SpaCy.

Below is the snapshot of how our app looks like:

![pos.gif](https://cdn.hashnode.com/res/hashnode/image/upload/v1604058531907/Bn0TSP8VL.gif)

### Table of Contents
1. Introduction to Natural Language Processing and Parts-of-Speech Tagging
2. Tools and Libraries
3. Pre-requisites
4. Code (Python Implementation)
5. Testing

### Introduction to Natural Language Processing and Parts-of-Speech Tagging
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

In this app, we'll dive into the concept of *Parts-of-Speech Tagging*.

#### Parts-of-Speech Tagging
*Parts-of-Speech Tagging* refers to assigning parts of speech (such as NOUN, ADJECTIVE, CONJUNCTION etc.) to individual words in a sentence, which means that it is performed at the token level.

POS tagging is useful, particularly when you have words that can have multiple POS tags. For instance, the word "promise" can be used as both a noun and verb, depending upon the context. Example-
- He could not keep the `promise (NOUN)` he had given to his mother. <br>
- I `promise (VERB)` my mother that I will be a good boy. 

 
While processing the natural language, it is extremely important to identify this difference. The spaCy library, which we have used in our app, comes with pre-built machine learning algorithms that is capable of returning the correct POS tag for the word, depending upon the context.

Now, let's take an example of parts-of-speech tagging:
In the sentence- *"I love to dance since childhood."*
> I -> Pronoun (PRON)<br>
love -> VERB<br>
to -> PART<br>
dance -> VERB<br>
since -> Subordinating Conjunction (SCONJ) <br>
childhood -> NOUN


Some universal POS tags are ([source](https://spacy.io/api/annotation#pos-tagging)) :

![pos tags.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1604065036972/X8ywqky-P.jpeg)

### Tools and Libraries
#### 1. SpaCy-
spaCy is known to be the fastest NLP framework in Python. It is easy to learn and use, and one can easily perform simple tasks using a few lines of code.

It provides a default model which can recognise correct parts-of-speech tags from a text, depending upon the context.

#### 2. Streamlit-
Streamlit is an open-source Python library that has been used in our app for the user interface. 
1. Make sure you have Python 3.6 or greater installed.
2. Install Streamlit using the command (in terminal):
```
pip install streamlit
```

### Pre-requisites
- Download and Install Python 3.6 or greater.
- Install Streamlit from the terminal using the command-
```
pip install streamlit 
``` 
- Install spacy and the model using the commands (on terminal)-
```
pip install spacy
python -m spacy download en_core_web_sm
```

### The code
Let's dive straight into the code-
```
#Core Pkgs
import streamlit as st

#NLP Pkgs
import spacy
nlp = spacy.load("en_core_web_sm") 
from spacy import displacy

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


#To extract the pos tags
def pos(raw_text):
	if raw_text != "":
		st.subheader("Part-of-speech Tags")
		# Process whole documents 
		doc = nlp(raw_text) 
		# Token and Tag 
		for token in doc: 
			st.write(token, token.pos_) 


#To visualize pos tags
def visualize(raw_text):
	if raw_text != "":
		doc = nlp(raw_text) 
		if "parser" in nlp.pipe_names:
		    st.subheader("Dependency Parse & Part-of-speech tags")
		    st.sidebar.header("Dependency Parse")
		    split_sents = st.sidebar.checkbox("Split sentences", value=True)
		    collapse_punct = st.sidebar.checkbox("Collapse punctuation", value=True)
		    collapse_phrases = st.sidebar.checkbox("Collapse phrases")
		    compact = st.sidebar.checkbox("Compact mode")
		    options = {
		        "collapse_punct": collapse_punct,
		        "collapse_phrases": collapse_phrases,
		        "compact": compact,
		    }
		    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
		    for sent in docs:
		        html = displacy.render(sent, options=options)
		        # Double newlines seem to mess with the rendering
		        html = html.replace("\n\n", "\n")
		        if split_sents and len(docs) > 1:
		            st.markdown(f"> {sent.text}")
		        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


def main():
	"""A Simple NLP-POS App"""
	st.title("Parts-of-Speech Tagging")
	st.markdown("Part-Of-Speech Tagging is the process by which we can tag each word of a sentence with its corresponding grammatical function (determinant, noun, adjective) using a mix of deep learning and probabilistic approach as in Named Entity Recognition.It also uses the library named as SpaCy.")

	raw_text = st.text_area("Your Text","")
	pos(raw_text)
	visualize(raw_text)
	
if __name__ == '__main__':
	main()
```
Breaking it down,
```
#Core Pkgs
import streamlit as st

#NLP Pkgs
import spacy
nlp = spacy.load("en_core_web_sm") 
from spacy import displacy
```
First and foremost task is to import the packages, libraries and frameworks to be used in our code. NLP Packages have been imported so that we can use the pre-trained model to extract the parts-of-speech tags. `nlp = spacy.load("en_core_web_sm") ` loads English tokenizer, tagger, parser, NER and word vectors.

```
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
```
Since we are using `displacy from spacy` to display the Parts-of-Speech in a nice html format on our front end (Graphical Representation), therefore, in order to render the POS in spacy using displacy, we'll wrap our result within an html as done in the above code snippet.

```
#To extract the pos tags
def pos(raw_text):
	if raw_text != "":
		st.subheader("Part-of-speech Tags")
		# Process whole documents 
		doc = nlp(raw_text) 
		# Token and Tag 
		for token in doc: 
			st.write(token, token.pos_) 

```
The above code snippet defines a function `pos(raw_text)` which extracts the tokens and the POS tags.

```
#To visualize pos tags
def visualize(raw_text):
	if raw_text != "":
		doc = nlp(raw_text) 
		if "parser" in nlp.pipe_names:
		    st.subheader("Dependency Parse & Part-of-speech tags")
		    st.sidebar.header("Dependency Parse")
		    split_sents = st.sidebar.checkbox("Split sentences", value=True)
		    collapse_punct = st.sidebar.checkbox("Collapse punctuation", value=True)
		    collapse_phrases = st.sidebar.checkbox("Collapse phrases")
		    compact = st.sidebar.checkbox("Compact mode")
		    options = {
		        "collapse_punct": collapse_punct,
		        "collapse_phrases": collapse_phrases,
		        "compact": compact,
		    }
		    
```
So here, we have defined a function `visualize(raw_text)` which is responsible for visualizing the POS tags in a graphical way. Using this function, we create the Dependency Parse. For this purpose, we have used displacy module from the spacy library. You can select various parameters like *Split Sentences*, *Collapse Punctuation*, *Collapse Phrases* and *Compact Mode* from the sidebar. 

```
		    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
		    for sent in docs:
		        html = displacy.render(sent, options=options)
		        # Double newlines seem to mess with the rendering
		        html = html.replace("\n\n", "\n")
		        if split_sents and len(docs) > 1:
		            st.markdown(f"> {sent.text}")
		        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
```
With this, we render our dependency parse using displacy and streamlit. We will set `unsafe_allow_html` to `True` for `st.write()` to render the html text as valid html.

```
def main():
	"""A Simple NLP-POS App"""
	st.title("Parts-of-Speech Tagging")
	st.markdown("Part-Of-Speech Tagging is the process by which we can tag each word of a sentence with its corresponding grammatical function (determinant, noun, adjective) using a mix of deep learning and probabilistic approach as in Named Entity Recognition.It also uses the library named as SpaCy.")

	raw_text = st.text_area("Your Text","")
	pos(raw_text)
	visualize(raw_text)
	
if __name__ == '__main__':
	main()
```
Last, we define `main()` function and provide a title ***Parts-of-Speech Tagging*** to our app. Then, we have given a small description about the app using `st.markdown()`.<br><br>
The user is then asked to enter the text. Then we call the `pos(raw_text)` and visualize(raw_text)` functions.<br><br>
Finally, we call our `main()` function.

### Test the app
To test the app, save the above python code with the name, say, `app.py`. Then, in the terminal, write-

```
streamlit run app.py
```

Thankyou for reading, I would love to connect with you at  [LinkedIn](https://www.linkedin.com/in/srishtii24/). 

