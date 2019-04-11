---
redirect_from:
  - "/features/notebooks/1-introduction-to-programming"
interact_link: content/features/notebooks/1_Introduction_to_Programming.ipynb
kernel_name: python3
title: 'Introduction to Python'
prev_page:
  url: /features/notebooks/0_Introduction_to_JupyterHub
  title: 'Getting Started'
next_page:
  url: /features/notebooks/2_Introduction_to_Dataframes_&_Plotting
  title: 'Introduction to Dataframes and Plotting'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Introduction to programming
_Written by Luke Chang_

In this notebook we will begin to learn how to use Python.  There are many different ways to install Python, but we recommend starting using Anaconda which is preconfigured for scientific computing.  Start with installing [Python 3.7](https://www.anaconda.com/distribution/).  For those who prefer a more configurable IDE, [Pycharm](https://www.jetbrains.com/pycharm/) is a nice option.  Python is a modular interpreted language with an intuitive minimal syntax that is quickly becoming one of the most popular languages for [conducting research](http://www.talyarkoni.org/blog/2013/11/18/the-homogenization-of-scientific-computing-or-why-python-is-steadily-eating-other-languages-lunch/).  You can use python for [stimulus presentation](http://www.psychopy.org/), [data analysis](http://statsmodels.sourceforge.net/), [machine-learning](http://scikit-learn.org/stable/), [scraping data](https://www.crummy.com/software/BeautifulSoup/), creating websites with [flask](http://flask.pocoo.org/) or [django](https://www.djangoproject.com/), or [neuroimaging data analysis](http://nipy.org/).

There are lots of free useful resources to learn how to use python and various modules.  See Yaroslav Halchenko's excellent [Dartmouth course](https://github.com/dartmouth-pbs/psyc161).  [Codeacademy](https://www.codecademy.com/) is a great interactive tutorial.  [Stack Overflow](http://stackoverflow.com/) is an incredibly useful resource for asking specific questions and seeing responses to others that have been rated by the development community. 

![image.png](attachment:image.png)

## Jupyter Notebooks
We will primarily be using [Jupyter Notebooks](http://jupyter.org/) to interface with Python.  A Jupyter notebook consists of **cells**. The two main types of cells you will use are code cells and markdown cells.

A **_code cell_** contains actual code that you want to run. You can specify a cell as a code cell using the pulldown menu in the toolbar in your Jupyter notebook. Otherwise, you can can hit esc and then y (denoted "esc, y") while a cell is selected to specify that it is a code cell. Note that you will have to hit enter after doing this to start editing it.
If you want to execute the code in a code cell, hit "shift + enter." Note that code cells are executed in the order you execute them. That is to say, the ordering of the cells for which you hit "shift + enter" is the order in which the code is executed. If you did not explicitly execute a cell early in the document, its results are now known to the Python interpreter.

**_Markdown cells_** contain text. The text is written in markdown, a lightweight markup language. You can read about its syntax [here](http://daringfireball.net/projects/markdown/syntax). Note that you can also insert HTML into markdown cells, and this will be rendered properly. As you are typing the contents of these cells, the results appear as text. Hitting "shift + enter" renders the text in the formatting you specify.  You can specify a cell as being a markdown cell in the Jupyter toolbar, or by hitting "esc, m" in the cell. Again, you have to hit enter after using the quick keys to bring the cell into edit mode.

In general, when you want to add a new cell, you can use the "Insert" pulldown menu from the Jupyter toolbar. The shortcut to insert a cell below is "esc, b" and to insert a cell above is "esc, a." Alternatively, you can execute a cell and automatically add a new one below it by hitting "alt + enter."



{:.input_area}
```python
print("Hello World")
```


## Package Management
Package managment in Python has been dramatically improving.  Anaconda has it's own package manager called 'conda'.  Use this if you would like to install a new module as it is optimized to work with anaconda. 

```
!conda install *package*
```

However, sometimes conda doesn't have a particular package.  In this case use the default python package manager called 'pip'. 

These commands can be run in your unix terminal or you can send them to the shell from a Jupyter notebook by starting the line with ```!```

It is easy to get help on how to use the package managers

```
!pip help install
```



{:.input_area}
```python
!pip help install
```




{:.input_area}
```python
!pip list --outdated
```




{:.input_area}
```python
!pip install setuptools --upgrade
```


## Variables
* Built-in
  * Numeric types:
    * **int**, **float**, **long**, complex
  * **str**ing 
  * **bool**ean
    * True / False
  * **NoneType**
* User defined

* Use the type() function to find the type for a value or variable

* Data can be converted using cast commands



{:.input_area}
```python
# Integer
a = 1
print(type(a))

# Float
b = 1.0
print(type(b))

# String
c = 'hello'
print(type(c))

# Boolean
d = True
print(type(d))

# None
e = None
print(type(e))

# Cast integer to string
print(type(str(a)))
```


## Math Operators
* +, -, *, and /
* Exponentiation **
* Modulo %

* Note that division with integers in Python 2.7 automatically rounds, which may not be intended.  It is recommended to import the division module from python3 `from __future__ import division`



{:.input_area}
```python
# Addition
a = 2 + 7
print(a)

# Subtraction
b = a - 5
print(b)

# Multiplication
print(b*2)

# Exponentiation
print(b**2)

# Modulo
print(4%9)

# Division
print(4/9)
```


## String Operators
* Some of the arithmetic operators also have meaning for strings. E.g. for string concatenation use `+` sign
* String repetition: Use `*` sign with a number of repetitions



{:.input_area}
```python
# Combine string
a = 'Hello'
b = 'World'
print(a + b)

# Repeat String
print(a*5)
```


## Logical Operators
Perform logical comparison and return Boolean value

```python
x == y # x is equal to y
x != y # x is not equal to y
x > y # x is greater than y
x < y # x is less than y
x >= y # x is greater than or equal to y 
x <= y # x is less than or equal to y```

|  X    | not X  | 
|-------|--------|
| True  | False  | 
| False | True   | 

|  X   | Y    | X AND Y | X OR Y | 
|------|------|---------|--------|
|True  | True | True  | True   | 
|True  | False| False | True   | 
|False | True | False | True   | 
|False | False| False | False  | 



{:.input_area}
```python
# Works for string
a = 'hello'
b = 'world'
c = 'Hello'
print(a==b)
print(a==c)
print(a!=b)

# Works for numeric
d = 5
e = 8
print(d < e)
```


## Conditional Logic (if...)
Unlike most other languages, Python uses tab formatting rather than closing conditional statements (e.g., end).

* Syntax:

``` python 
if condition: 
    do something
```

* Implicit conversion of the value to bool() happens if `condition` is of a different type than **bool**, thus all of the following should work:

```python
if condition:
    do_something
elif condition:
    do_alternative1
else:
    do_otherwise # often reserved to report an error
                 # after a long list of options
    ```




{:.input_area}
```python
n = 1

if n:
    print("n is non-0")

if n is None:
    print("n is None")
    
if n is not None:
    print("n is not None")
```


## Loops
* **for** loop is probably the most popular loop construct in Python:

```python
for target in sequence:
    do_statements
```

* However, it's also possible to use a **while** loop to repeat statements while `condition` remains True:

```python
while condition do:
    do_statements
```



{:.input_area}
```python
string = "Python is going to make conducting research easier"
for c in string:
    print(c)

```




{:.input_area}
```python
x = 0
end = 10

csum = 0
while x < end:
    csum += x
    print(x, csum)
    x += 1
print("Exited with x==%d" % x )
```


## Functions
A **function** is a named sequence of statements that performs a computation.  You define the function by giving it a name, specify a sequence of statements, and optionally values to return.  Later, you can “call” the function by name. 
```python
def make_upper_case(text):
    return (text.upper())
```
* The expression in the parenthesis is the **argument**. 
* It is common to say that a function **“takes” an argument** and **“returns” a result**.
* The result is called the **return value**.

The first line of the function definition is called the **header**; the rest is called the **body**.

The header has to end with a colon and the body has to be indented.
It is a common practice to use 4 spaces for indentation, and to avoid mixing with tabs.

Function body in Python ends whenever statement begins at the original level of indentation.  There is no **end** or **fed** or any other identify to signal the end of function.  Indentation is part of the the language syntax in Python, making it more readable and less cluttered.



{:.input_area}
```python
def make_upper_case(text):
    return (text.upper())

string = "Python is going to make conducting research easier"

print(make_upper_case(string))
```


## Python Containers
There are 4 main types of builtin containers for storing data in Python:
* list
* tuple
* dict
* set

### Lists
In Python, a list is a mutable sequence of values.  Mutable means that we can change separate entries within a list. For a more in depth tutorial on lists look [here](http://nbviewer.jupyter.org/github/dartmouth-pbs/psyc161/blob/master/classes/02d-Python-Fundamentals-Containers-Lists.ipynb)

* Each value in the list is an element or item
* Elements can be any Python data type
* Lists can mix data types

* Lists are initialized with ```[]``` or ```list()```
```python
l = [1,2,3]
```
* 
Elements within a list are indexed (**starting with 0**)
```python
l[0]
```

* 
Elements can be nested lists
```python
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

* 
Lists can be *sliced*.
```python
l[start:stop:stride]
```

* Like all python containers, lists have many useful methods that can be applied
```python
a.insert(index,new element)
a.append(element to add at end)
len(a)
```

* 
List comprehension is a *Very* powerful technique allowing for efficient construction of new lists.
```python
[a for a in l]
```




{:.input_area}
```python
# Indexing and Slicing
a = ['lists','are','arrays']
print(a[0])
print(a[1:3])

# List methods
a.insert(2,'python')
a.append('.')
print(a)
print(len(a))

# List Comprehension
print([x.upper() for x in a])
```


### Dictionaries

* In Python, a dictionary (or `dict`) is mapping between a set of
indices (**keys**) and a set of **values**

* The items in a dictionary are key-value pairs

* Keys can be any Python data type

* Dictionaries are unordered

* Here is a more indepth tutorial on [dictionaries](http://nbviewer.jupyter.org/github/dartmouth-pbs/psyc161/blob/master/classes/03c-Python-Fundamentals-Containers-Dicts.ipynb)



{:.input_area}
```python
# Dictionaries
eng2sp = {}
eng2sp['one'] = 'uno'
print(eng2sp)

eng2sp = {'one': 'uno', 'two': 'dos', 'three': 'tres'}
print(eng2sp)

print(eng2sp.keys())
print(eng2sp.values())
```


### Tuples
In Python, a **tuple** is an immutable sequence of values, meaning they can't be changed

* Each value in the tuple is an element or item

* Elements can be any Python data type

* Tuples can mix data types

* Elements can be nested tuples

* **Essentially tuples are immutable lists**

Here is a nice tutorial on [tuples](http://nbviewer.jupyter.org/github/dartmouth-pbs/psyc161/blob/master/classes/03b-Python-Fundamentals-Containers-Tuples.ipynb)



{:.input_area}
```python
numbers = (1, 2, 3, 4)
print(numbers)

t2 = 1, 2
print(t2)
```


## sets
In Python, a `set` is an efficient storage for "membership" checking

* `set` is like a `dict` but only with keys and without values

* a `set` can also perform set operations (e.g., union intersection)

* Here is more info on [sets](http://nbviewer.jupyter.org/github/dartmouth-pbs/psyc161/blob/master/classes/03d-Python-Fundamentals-Containers-Sets.ipynb)




{:.input_area}
```python
# Union
print({1, 2, 3, 'mom', 'dad'} | {2, 3, 10})

# Intersection
print({1, 2, 3, 'mom', 'dad'} & {2, 3, 10})

# Difference
print({1, 2, 3, 'mom', 'dad'} - {2, 3, 10})
```


## Modules
A *Module* is a python file that contains a collection of related definitions. Python has *hundreds* of standard modules.  These are organized into what is known as the [Python Standard Library](http://docs.python.org/library/).  You can also create and use your own modules.  To use functionality from a module, you first have to import the entire module or parts of it into your namespace

* To import the entire module, use 

  ```python
  import module_name```

* You can also import a module using a specific name 

  ```python
  import module_name as new_module_name```

* To import specific definitions (e.g. functions, variables, etc) from the module into your local namespace, use

```python
from module_name import name1, name2
```
   which will make those available directly in your ```namespace```



{:.input_area}
```python
import os
from glob import glob
```


Here let's try and get the path of the current working directory using functions from the `os` module



{:.input_area}
```python
os.path.abspath(os.path.curdir)
```


It looks like we are currently in the notebooks folder of the github repository.  Let's use glob, a pattern matching function, to list all of the csv files in the Data folder.  



{:.input_area}
```python
data_file_list = glob(os.path.join('..','Data','*csv'))
print(data_file_list)
```


This gives us a list of the files including the relative path from the current directory.  What if we wanted just the filenames?  There are several different ways to do this.  First, we can use the the `os.path.basename` function.  We loop over every file, grab the base file name and then append it to a new list. 



{:.input_area}
```python
file_list = []
for f in data_file_list:
    file_list.append(os.path.basename(f))

print(file_list)
```


Alternatively, we could loop over all files and split on the `/` character.  This will create a new list where each element is whatever characters are separated by the splitting character.  We can then take the last element of each list.



{:.input_area}
```python
file_list = []
for f in data_file_list:
    file_list.append(f.split('/')[-1])

print(file_list)
```


It is also sometimes even cleaner to do this as a list comprehension



{:.input_area}
```python
[os.path.basename(x) for x in data_file_list]
```


# Exercises

### Find Even Numbers
Let’s say I give you a list saved in a variable: a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]. Make a new list that has only the even elements of this list in it.



{:.input_area}
```python

a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

b = [bool(1-x%2) for x in a]

c = []
for x in a:
    if not x%2:
        c.append(x)
        
def evens_only(seq):
    return [n for n in seq if n%2 == 0]

evens_only(a)
```





{:.output .output_data_text}
```
[4, 16, 36, 64, 100]
```



### Find Maximal Range
Given an array length 1 or more of ints, return the difference between the largest and smallest values in the array. 



{:.input_area}
```python
import numpy as np

a = np.random.randint(0,100, 10)
max(a)-min(a)
```





{:.output .output_data_text}
```
93
```



### Duplicated Numbers
Find the numbers in list a that are also in list b

a = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361]

b = [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]



{:.input_area}
```python
a = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361]

b = [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]

print(set(a) & set(b))

print([x for x in a if x in b])
```


{:.output .output_stream}
```
{0, 256, 64, 4, 36, 100, 196, 324, 16, 144}
[0, 4, 16, 36, 64, 100, 144, 196, 256, 324]

```

### Speeding Ticket Fine
You are driving a little too fast on the highway, and a police officer stops you. Write a function that takes the speed as an input and returns the fine.  Your function must use a dictionary.


If speed is 60 or less, the result is `$0`. If speed is between 61 and 80 inclusive, the result is `$100`. If speed is 81 or more, the result is `$500`. 




{:.input_area}
```python
def speeding_ticket(speed):
    if speed <= 60:
        return '$0'
    elif (speed >=61) & (speed < 100):
        return '$100'
    else:
        return ('$500')
speeding_ticket(81)
```





{:.output .output_data_text}
```
'$100'
```


