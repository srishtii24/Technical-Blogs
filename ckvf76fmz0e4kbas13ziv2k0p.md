## Very Useful Built-in Filters in Django Template Language

### What are Filters in DTL?
Before learning about the most important filters in Django Template Language (DTL), let's learn about ** What Filters are!**

> While  [tags](https://srishti.hashnode.dev/most-important-tags-in-django-template-language) are enclosed in curly braces and a percent, i.e., `{% tag %}`, and variables are enclosed in double curly braces, i.e., `{{ variable }}`. **Variables can be influenced, modified or filtered using FILTERS**.

1. Filters in DTL are the functions registered to the template engine which takes a value as input and returns transformed value.
2. To apply filters on a variable, we need to use a **pipe (|) symbol** in that variable. *For example:* `{{ name|lower }}` : This filter will modify the variable value in lowercase. Here, `name` is a variable, `|` is the pipe symbol and `lower` is the django filter.
3. We can use more than one filter on the variable. *For example: * `{{variable|filter1|filter2}}`.
4. All the default filter functions are written in a file `defaultfilters.py`. This is how a `lower` filter looks like:
```
@register.filter(is_safe=True)
@stringfilter
def lower(value):
    """Convert a string into all lowercase."""
    return value.lower()
``` 

### Important Filters
**1. `default`**:
> If a variable doesn't contain anything, then default value is displayed.

*For example-*

```
{{ value|default:"Unknown" }}
```
 
 **2. `length`**:
> It returns the length of the value. It works for both lists and strings.

*For example-* If value is `['a', 'b', 'c', 'd']`

```
{{ value|length }}
``` 
The above snippet returns `4`.

**3. `capfirst`**:
> It is used to capitalize the first character of the value.  If the first character is not a letter, this filter has no effect.

*For example-* If value is `i code In djANgO`

```
{{ value|capfirst }}
``` 
The above snippet returns `I code in Django`.

**4. `first`**:
> It is used to return the first item of the list.

*For example-* If value is `['a', 'b', 'c', 'd']`

```
{{ value|first}}
``` 
The above snippet returns the first item in list ,i.e., `a`.

**5. `upper`**:
> As the name suggests, it is used to convert the string into uppercase .

*For example-* If value is `i code In djANgO`

```
{{ value|upper}}
``` 
The above snippet returns all characters in variable value in uppercase ,i.e., `I CODE IN DJANGO`.

**6. `lower`**:
> As the name suggests, it is used to convert the string into lowercase .

*For example-* If value is `i code In djANgO`

```
{{ value|lower}}
``` 
The above snippet returns all characters in variable value in lowercase ,i.e., `i code in django`.

**7. `title`**:
> As the name suggests, it is used to convert the string into title case by making words start with an uppercase character and the remaining characters lowercase.

*For example-* If value is `i code In djANgO`

```
{{ value|title}}
``` 
The above snippet returns all characters in variable value in titlecase ,i.e., `I Code In Django`.

**8. `date`**:
> It is used to include only date if we have datetime variable. Also, it formats a date according to the given format.

*For example-*

```
{{ value|date:"D d M Y" }}
``` 
If value is a datetime object (e.g., the result of `datetime.datetime.now()`), the output will be the string `Sun 31 Oct 2021`.

> For all the possible date formats, refer the [documentation](https://docs.djangoproject.com/en/3.1/ref/templates/builtins/#date).
> In the same way, you set the [time](https://docs.djangoproject.com/en/3.1/ref/templates/builtins/#time) format. You can also combine it to show both.

**9. `cut`**:
> It removes all values of arg from the given string.

*For example-*
```
{{ value|cut:" " }}
```
If value is "String with spaces", the output will be "Stringwithspaces".

**10. `join`**:
> It joins a list with a string, like Python’s str.join(list)

*For example-*
```
{{ value|join:" // " }}
```
If value is the list ['a', 'b', 'c'], the output will be the string `a // b // c`.

**11. `truncatechars`**
> It truncates a string if it is longer than the specified number of characters. Truncated strings will end with a translatable ellipsis character (“…”).

*For example-*
```
{{ value|truncatechars:7 }}
```
If value is "Joel is a slug", the output will be "Joel i…".


Thankyou for reading, I would love to connect with you at  [LinkedIn](https://www.linkedin.com/in/srishtii24/) .