## Sentiment Analysis app using Python, Flair and Streamlit

This blog article covers the implementation of "Sentiment Analysis app" using Python, Flair and Streamlit. This is how our app looks like: 

![sentiment.gif](https://cdn.hashnode.com/res/hashnode/image/upload/v1604337657893/5QrFG9p3D.gif)

### Table of Contents
1. Introduction 
2. Tools and Libraries
3. Pre-requisites
4. Structure and workflow of the app
5. Code (Python Implementation)
6. Testing

### Introduction
#### Natural Language Processing-
Natural language is the language used by humans to communicate with each other (verbally or textually - for instance, face-to-face conversations, blogs, emails, SMS messages, etc.). Natural language is incredibly important for computers to understand for some reasons as under:

- It can be viewed as a source of huge amount of data, and if processed intelligently by the system, can yield the useful information.
- It can be helpful for computers to better communicate with humans.

However, unlike humans, computers can't easily comprehend natural language. Sophisticated methods and techniques are needed to translate natural language into a format understandable by computers. ***Natural Language Processing*** (NLP) helps us achieve this objective. 

#### Sentiment Analysis
Sentiment Analysis from a text is a classical problem of NLP. It is used when you try to predict the sentiment of customer feedback in a restaurant, a shopping site, etc. This app is a text analysis technique that detects the sentiment of the user, whether it's positive or negative.

This technique allows brands or the companies to listen attentively to their customers by examining their feedback, and tailoring products and services to meet their needs.

For instance, using sentiment analysis to automatically analyse 4,000+ reviews about your product can help you discover if customers are happy about your service and pricing plans or not. 

### Tools and Libraries
#### 1. Flair-
It is a Natural Language Processing (NLP) library developed, and open-sourced by Zalando Research. It’s framework is built on PyTorch, one of the best deep learning frameworks out there. 

There are numerous features packaged into the Flair library. Some of them are:

- It consists of popular and state-of-the-art word embeddings, such as BERT, GloVe, ELMo, Character Embeddings, etc. 
- Flair’s interface allows us to combine various word embeddings and then use them to embed documents. 
- Flair supports a number of languages – and is always looking to add new ones.

To install Flair, you'll need **Python 3.6** or greater.


#### 2. Streamlit-
Streamlit is an open-source Python library that has been used in our app for the user interface. 


### Pre-requisites
- Download and Install Python 3.6 or greater.
- Install Streamlit from the terminal using the command-
```
pip install streamlit 
``` 
- Install Flair from the terminal using the command-
```
pip install flair
```
This will install all the concerned packages needed to run Flair. This also include PyTorch on the top of which Flair sits.


### Structure and workflow of the app
The app asks for a text from the user and then the text is analysed for its polarity using Flair.

### The code
Let's dive straight into the code-
```
import streamlit as st
from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np
global tagger

def load_flair():
	return TextClassifier.load('en-sentiment')

def main():
	tagger = load_flair()

	st.markdown("<h1 style = 'textalign:center; color:blue;'> Sentiment Detection </h1>", unsafe_allow_html = True)
	st.write("Sentiment Detection from text is a classical problem. This is used when you try to predict the sentiment of comments on a restaurant. This app analyzes the sentiment of the user, whether it's Postive or Negative.")

	input_sent = st.text_input("Input Sentence", "Although not well rated, the food in this restaurant was tasty and I enjoyed the meal!")

	s = Sentence(input_sent)
	tagger.predict(s)
	st.write("### Your Sentence is ", str(s.labels))

if __name__ == '__main__':
	main()
```

Breaking it down,
```
import streamlit as st
from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np
global tagger
```
So here, we've imported the packages, libraries and frameworks to be used in our code. Also, we've assigned `tagger` as the **global** variable

```
def load_flair():
	return TextClassifier.load('en-sentiment')
```
Here, we've defined a function `load_flair()` to load a pre-trained sentiment analysis model.

```
def main():
	tagger = load_flair()

	st.markdown("<h1 style = 'textalign:center; color:blue;'> Sentiment Detection </h1>", unsafe_allow_html = True)
	st.write("Sentiment Detection from text is a classical problem. This is used when you try to predict the sentiment of comments on a restaurant. This app analyzes the sentiment of the user, whether it's Postive or Negative.")

	input_sent = st.text_input("Input Sentence", "Although not well rated, the food in this restaurant was tasty and I enjoyed the meal!")

	s = Sentence(input_sent)
	tagger.predict(s)
	st.write("### Your Sentence is ", str(s.labels))

if __name__ == '__main__':
	main()
```
Then we've defined our `main()` function which gives the title on the top and a little description about the app using `st.markdown()` and `st.write()`. Next, the user is prompted to enter the text. We've even provided a default text. The text is then analysed using the loaded model and returns the polarity of the text. Then, we've called the `main()` function.

### Test the app
To test the app, save the above python code with the name, say, `app.py`. Then, in the terminal, write-

```
streamlit run app.py
```

Thankyou for reading, I would love to connect with you at  [LinkedIn](https://www.linkedin.com/in/srishtii24/). <br><br>



