## Most Important {% Tags %} in Django Template Language

Before learning the important tags in Django Template, let's learn about ***What are Tags in Django.***


> Tags are the most important features in django template. Tags look like: {% tag %}.

**Features:**
1. Some tags are used to **perform loops or logic **.
2. Some tags are used to **load external information** into the templates.

*Note:* Some tags require **beginning and ending tags**

### **Important Tags:**
#### 1. for 
> **Syntax: ** `{% for %}` --------- `{% endfor %}`

If you want to use a **looping over each items**, then you can use a **`{% for %}`** tag to loop every items.

*For example:* To display a list of fruit names
```
<ul>
    {% for fruit in fruits_list %}
        <li> {{ fruit.name }} </li>
    {% endfor %}
</ul>
``` 
Here, `fruits_list` is a list of dictionaries: `[{'name': 'apple', 'quantity':4},{'name': 'kiwi', 'quantity':2},{'name': 'orange', 'quantity':10}]`.


#### 2. if, elif, else
> **Syntax: ** `{% if %}` --- `{% elif %}` --- `{% else %}` --- `{% endif %}`

If that variable is "True", the contents of that block are displayed.

*For example:* 
```
    {% if fruits_list %}
        Number of fruits: {{ fruits_list|length }} 
    {% elif vegetables_list %}
        Number of Vegetables: {{ vegetables_list|length }} 
    {% else %}
        Buy fruits or vegetables. 
    {% endif %}
``` 
#### 3. comment
> **Syntax: ** `{% comment %}` --------- `{% endcomment %}`

If you want to **comment something**, you can use **comment tag** and in between the opening and closing tags, write the comment.

*For example:* 
```
    {% comment "Optional note" %}
          This part will not be displayed.
           Both single and multi-line comments works!
    {% endcomment %}
``` 
*Note:* comment tags can't be nested.

#### 4. csrf_token
> **Syntax: ** `{% csrf_token %}` 

It is the most important tag in templates language. **`csrf_token` protects** our web app from **Cross Site Request Forgery Attacks**.

*For example:* Paste this tag in between form tag
```
    <form method="POST">
        {%csrf_token %}
    </form>
``` 

#### 5. extends
> **Syntax: ** `{% extends %}` 

Signals that current template **extends a parent template**.

*For example:* 
```
    {% extends "base.html" %}
``` 

#### 6. include
> **Syntax: ** `{% include %}` 

It is basically used to **load other templates**.

*For example:* With this tag, we can load content of other template in current template
```
    {% include "actionforms.html" %}
``` 

#### 7. load
> **Syntax: ** `{% load %}` 

It is basically used to **load custom templates tag set**.

*For example:* Here, we are loading static in our template so we can use static tag
```
    {% load static %}
``` 

#### 8. url
> **Syntax: ** `{% url %}` 

It is basically used to **return an absolute path reference (a URL without the domain name)**.

*For example:* 
```
    <a href="{% url 'some-url-name' v1 v2 %}"></a>
``` 
Thankyou for reading, I would love to connect with you at [LinkedIn](https://www.linkedin.com/in/srishtii24/). 