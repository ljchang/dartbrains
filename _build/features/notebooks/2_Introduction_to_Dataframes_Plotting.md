---
redirect_from:
  - "/features/notebooks/2-introduction-to-dataframes-plotting"
interact_link: content/features/notebooks/2_Introduction_to_Dataframes_Plotting.ipynb
kernel_name: python3
title: 'Introduction to Dataframes and Plotting'
prev_page:
  url: /features/notebooks/1_Introduction_to_Programming
  title: 'Introduction to Python'
next_page:
  url: /features/notebooks/3_Introduction_to_NeuroimagingData_in_Python
  title: 'Introduction to Neuroimaging Data'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Dataframes ( Pandas ) and Plotting ( Matplotlib/Seaborn )

*Written by Jin Cheong & Luke Chang*

In this lab we are going to learn how to load and manipulate datasets in a dataframe format using Pandas   
and create beautiful plots using Matplotlib and Seaborn. Pandas is akin to a data frame in R and provides an intuitive way to interact with data in a 2D data frame. Matplotlib is a standard plotting library that is similar in functionality to Matlab's object oriented plotting. Seaborn is also a plotting library built on the Matplotlib framework which carries useful pre-configured plotting schemes. 

After the tutorial you will have the chance to apply the methods to a new set of data. 

Also, [here is a great set of notebooks](https://github.com/jakevdp/PythonDataScienceHandbook) that also covers the topic.  In addition, here is a brief [video](https://neurohackademy.org/course/complex-data-structures/) providing a useful introduction to pandas. 


First, we load the basic packages we will be using in this tutorial.  Notice how we import the modules using an abbreviated name.  This is to reduce the amount of text we type when we use the functions.

**Note**: `%matplotlib inline` is an example of 'cell magic' and enables plotting *within* the notebook and not opening a separate window. In addition, you may want to try using `%matplotlib notebook`, which will allow more interactive plotting.



{:.input_area}
```python
%matplotlib inline 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


# Pandas

## Loading Data
We use the pd.read_csv() to load a .csv file into a dataframe. 
Note that read_csv() has many options that can be used to make sure you load the data correctly. 

The command `pwd` will print the path of the current working directory.



{:.input_area}
```python
pwd
```





{:.output .output_data_text}
```
'/Users/lukechang/Github/dartbrains/content/features/notebooks'
```



Pandas has many ways to read data different data formats into a dataframe.  Here we will use the `pd.read_csv` function.



{:.input_area}
```python
df = pd.read_csv('../../data/salary.csv', sep = ',', header='infer')
```


You can always use the `?` to access the docstring for a function for more information about the inputs and general useage guidelines.



{:.input_area}
```python
pd.read_csv?
```


## Ways to check the dataframe
There are many ways to examine your dataframe. One easy way is to execute the dataframe itself. 



{:.input_area}
```python
df
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64451</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>6</th>
      <td>64366</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59344</td>
      <td>0</td>
      <td>bio</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>58560</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>58294</td>
      <td>0</td>
      <td>bio</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>56092</td>
      <td>0</td>
      <td>bio</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54452</td>
      <td>0</td>
      <td>bio</td>
      <td>13.0</td>
      <td>43.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54269</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>55125</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>14</th>
      <td>97630</td>
      <td>0</td>
      <td>chem</td>
      <td>34.0</td>
      <td>64.0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>15</th>
      <td>82444</td>
      <td>0</td>
      <td>chem</td>
      <td>31.0</td>
      <td>61.0</td>
      <td>42</td>
    </tr>
    <tr>
      <th>16</th>
      <td>76291</td>
      <td>0</td>
      <td>chem</td>
      <td>29.0</td>
      <td>65.0</td>
      <td>33</td>
    </tr>
    <tr>
      <th>17</th>
      <td>75382</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>18</th>
      <td>64762</td>
      <td>0</td>
      <td>chem</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>29</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62607</td>
      <td>0</td>
      <td>chem</td>
      <td>20.0</td>
      <td>45.0</td>
      <td>34</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60373</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>21</th>
      <td>58892</td>
      <td>0</td>
      <td>chem</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>47021</td>
      <td>0</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>23</th>
      <td>44687</td>
      <td>0</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>24</th>
      <td>104828</td>
      <td>0</td>
      <td>geol</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>25</th>
      <td>71456</td>
      <td>0</td>
      <td>geol</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>26</th>
      <td>65144</td>
      <td>0</td>
      <td>geol</td>
      <td>7.0</td>
      <td>37.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>27</th>
      <td>52766</td>
      <td>0</td>
      <td>geol</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>28</th>
      <td>112800</td>
      <td>0</td>
      <td>neuro</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>33</td>
    </tr>
    <tr>
      <th>29</th>
      <td>105761</td>
      <td>0</td>
      <td>neuro</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>65285</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>48</th>
      <td>62557</td>
      <td>0</td>
      <td>stat</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>49</th>
      <td>61947</td>
      <td>0</td>
      <td>stat</td>
      <td>22.0</td>
      <td>58.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>50</th>
      <td>58565</td>
      <td>0</td>
      <td>stat</td>
      <td>29.0</td>
      <td>59.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>51</th>
      <td>58365</td>
      <td>0</td>
      <td>stat</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53656</td>
      <td>0</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>53</th>
      <td>51391</td>
      <td>0</td>
      <td>stat</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>54</th>
      <td>96936</td>
      <td>0</td>
      <td>physics</td>
      <td>15.0</td>
      <td>50.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>55</th>
      <td>83216</td>
      <td>0</td>
      <td>physics</td>
      <td>11.0</td>
      <td>37.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>56</th>
      <td>72044</td>
      <td>0</td>
      <td>physics</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>57</th>
      <td>64048</td>
      <td>0</td>
      <td>physics</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58888</td>
      <td>0</td>
      <td>physics</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>59</th>
      <td>58744</td>
      <td>0</td>
      <td>physics</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>60</th>
      <td>55944</td>
      <td>0</td>
      <td>physics</td>
      <td>21.0</td>
      <td>51.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>61</th>
      <td>54076</td>
      <td>0</td>
      <td>physics</td>
      <td>19.0</td>
      <td>49.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>62</th>
      <td>82142</td>
      <td>0</td>
      <td>math</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>63</th>
      <td>70509</td>
      <td>0</td>
      <td>math</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>64</th>
      <td>60320</td>
      <td>0</td>
      <td>math</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>65</th>
      <td>55814</td>
      <td>0</td>
      <td>math</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>66</th>
      <td>53638</td>
      <td>0</td>
      <td>math</td>
      <td>4.0</td>
      <td>42.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>67</th>
      <td>53517</td>
      <td>2</td>
      <td>math</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>68</th>
      <td>59139</td>
      <td>1</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>69</th>
      <td>52968</td>
      <td>1</td>
      <td>bio</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>70</th>
      <td>55949</td>
      <td>1</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>71</th>
      <td>58893</td>
      <td>1</td>
      <td>neuro</td>
      <td>10.0</td>
      <td>35.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>72</th>
      <td>53662</td>
      <td>1</td>
      <td>neuro</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>73</th>
      <td>57185</td>
      <td>1</td>
      <td>stat</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>74</th>
      <td>52254</td>
      <td>1</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>75</th>
      <td>61885</td>
      <td>1</td>
      <td>math</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>76</th>
      <td>49542</td>
      <td>1</td>
      <td>math</td>
      <td>3.0</td>
      <td>33.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 6 columns</p>
</div>
</div>



However, often the dataframes can be large and we may be only interested in seeing the first few rows.  `df.head()` is useful for this purpose.  `shape` is another useful method for getting the dimensions of the matrix.  We will print the number of rows and columns in this data set by using output formatting.  Use the `%` sign to indicate the type of data (e.g., `%i`=integer, `%d`=float, `%s`=string), then use the `%` followed by a tuple of the values you would like to insert into the text.  See [here](https://pyformat.info/) for more info about formatting text.



{:.input_area}
```python
print('There are %i rows and %i columns in this data set' % df.shape) 
```


{:.output .output_stream}
```
There are 77 rows and 6 columns in this data set

```



{:.input_area}
```python
df.head()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>
</div>



On the top row, you have column names, that can be called like a dictionary (a dataframe can be essentially thought of as a dictionary with column names as the keys). The left most column (0,1,2,3,4...) is called the index of the dataframe. The default index is sequential integers, but it can be set to anything as long as each row is unique (e.g., subject IDs)



{:.input_area}
```python
print("Indexes")
print(df.index)
print("Columns")
print(df.columns)
print("Columns are like keys of a dictionary")
print(df.keys())
```


{:.output .output_stream}
```
Indexes
RangeIndex(start=0, stop=77, step=1)
Columns
Index(['salary', 'gender', 'departm', 'years', 'age', 'publications'], dtype='object')
Columns are like keys of a dictionary
Index(['salary', 'gender', 'departm', 'years', 'age', 'publications'], dtype='object')

```

You can access the values of a column by calling it directly. Double bracket returns a dataframe




{:.input_area}
```python
df[['salary']]
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64451</td>
    </tr>
    <tr>
      <th>6</th>
      <td>64366</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59344</td>
    </tr>
    <tr>
      <th>8</th>
      <td>58560</td>
    </tr>
    <tr>
      <th>9</th>
      <td>58294</td>
    </tr>
    <tr>
      <th>10</th>
      <td>56092</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54452</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54269</td>
    </tr>
    <tr>
      <th>13</th>
      <td>55125</td>
    </tr>
    <tr>
      <th>14</th>
      <td>97630</td>
    </tr>
    <tr>
      <th>15</th>
      <td>82444</td>
    </tr>
    <tr>
      <th>16</th>
      <td>76291</td>
    </tr>
    <tr>
      <th>17</th>
      <td>75382</td>
    </tr>
    <tr>
      <th>18</th>
      <td>64762</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62607</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60373</td>
    </tr>
    <tr>
      <th>21</th>
      <td>58892</td>
    </tr>
    <tr>
      <th>22</th>
      <td>47021</td>
    </tr>
    <tr>
      <th>23</th>
      <td>44687</td>
    </tr>
    <tr>
      <th>24</th>
      <td>104828</td>
    </tr>
    <tr>
      <th>25</th>
      <td>71456</td>
    </tr>
    <tr>
      <th>26</th>
      <td>65144</td>
    </tr>
    <tr>
      <th>27</th>
      <td>52766</td>
    </tr>
    <tr>
      <th>28</th>
      <td>112800</td>
    </tr>
    <tr>
      <th>29</th>
      <td>105761</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>65285</td>
    </tr>
    <tr>
      <th>48</th>
      <td>62557</td>
    </tr>
    <tr>
      <th>49</th>
      <td>61947</td>
    </tr>
    <tr>
      <th>50</th>
      <td>58565</td>
    </tr>
    <tr>
      <th>51</th>
      <td>58365</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53656</td>
    </tr>
    <tr>
      <th>53</th>
      <td>51391</td>
    </tr>
    <tr>
      <th>54</th>
      <td>96936</td>
    </tr>
    <tr>
      <th>55</th>
      <td>83216</td>
    </tr>
    <tr>
      <th>56</th>
      <td>72044</td>
    </tr>
    <tr>
      <th>57</th>
      <td>64048</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58888</td>
    </tr>
    <tr>
      <th>59</th>
      <td>58744</td>
    </tr>
    <tr>
      <th>60</th>
      <td>55944</td>
    </tr>
    <tr>
      <th>61</th>
      <td>54076</td>
    </tr>
    <tr>
      <th>62</th>
      <td>82142</td>
    </tr>
    <tr>
      <th>63</th>
      <td>70509</td>
    </tr>
    <tr>
      <th>64</th>
      <td>60320</td>
    </tr>
    <tr>
      <th>65</th>
      <td>55814</td>
    </tr>
    <tr>
      <th>66</th>
      <td>53638</td>
    </tr>
    <tr>
      <th>67</th>
      <td>53517</td>
    </tr>
    <tr>
      <th>68</th>
      <td>59139</td>
    </tr>
    <tr>
      <th>69</th>
      <td>52968</td>
    </tr>
    <tr>
      <th>70</th>
      <td>55949</td>
    </tr>
    <tr>
      <th>71</th>
      <td>58893</td>
    </tr>
    <tr>
      <th>72</th>
      <td>53662</td>
    </tr>
    <tr>
      <th>73</th>
      <td>57185</td>
    </tr>
    <tr>
      <th>74</th>
      <td>52254</td>
    </tr>
    <tr>
      <th>75</th>
      <td>61885</td>
    </tr>
    <tr>
      <th>76</th>
      <td>49542</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 1 columns</p>
</div>
</div>



Single bracket returns a Series



{:.input_area}
```python
df['salary']
```





{:.output .output_data_text}
```
0      86285
1      77125
2      71922
3      70499
4      66624
5      64451
6      64366
7      59344
8      58560
9      58294
10     56092
11     54452
12     54269
13     55125
14     97630
15     82444
16     76291
17     75382
18     64762
19     62607
20     60373
21     58892
22     47021
23     44687
24    104828
25     71456
26     65144
27     52766
28    112800
29    105761
       ...  
47     65285
48     62557
49     61947
50     58565
51     58365
52     53656
53     51391
54     96936
55     83216
56     72044
57     64048
58     58888
59     58744
60     55944
61     54076
62     82142
63     70509
64     60320
65     55814
66     53638
67     53517
68     59139
69     52968
70     55949
71     58893
72     53662
73     57185
74     52254
75     61885
76     49542
Name: salary, Length: 77, dtype: int64
```



You can also call a column like an attribute if the column name is a string 




{:.input_area}
```python
df.salary
```





{:.output .output_data_text}
```
0      86285
1      77125
2      71922
3      70499
4      66624
5      64451
6      64366
7      59344
8      58560
9      58294
10     56092
11     54452
12     54269
13     55125
14     97630
15     82444
16     76291
17     75382
18     64762
19     62607
20     60373
21     58892
22     47021
23     44687
24    104828
25     71456
26     65144
27     52766
28    112800
29    105761
       ...  
47     65285
48     62557
49     61947
50     58565
51     58365
52     53656
53     51391
54     96936
55     83216
56     72044
57     64048
58     58888
59     58744
60     55944
61     54076
62     82142
63     70509
64     60320
65     55814
66     53638
67     53517
68     59139
69     52968
70     55949
71     58893
72     53662
73     57185
74     52254
75     61885
76     49542
Name: salary, Length: 77, dtype: int64
```



You can create new columns to fit your needs. 
For instance you can set initialize a new column with zeros. 



{:.input_area}
```python
df['pubperyear'] = 0
```


Here we can create a new column pubperyear, which is the ratio of the number of papers published per year



{:.input_area}
```python
df['pubperyear'] = df['publications']/df['years']
```




{:.input_area}
```python
df.head()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
      <td>2.769231</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
      <td>1.535714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
      <td>2.090909</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Indexing and slicing
Indexing in Pandas can be tricky. There are four ways to index: loc, iloc, and explicit indexing(useful for booleans).  


First, we will try using `.loc`.  This method references the explicit index. it works for both index names and also column names.



{:.input_area}
```python
df.loc[0, ['salary']]
```





{:.output .output_data_text}
```
salary    86285
Name: 0, dtype: object
```



Next we wil try `.iloc`.  This method references the implicit python index (starting from 0, exclusive of last number).  You can think of this like row by column indexing using integers.



{:.input_area}
```python
df.iloc[0:3, 0:3]
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Let's make a new data frame with just Males and another for just Females. Notice, how we added the `.reset_index(drop=True)` method?   This is because assigning a new dataframe based on indexing another dataframe will retain the *original* index.  We need to explicitly tell pandas to reset the index if we want it to start from zero.



{:.input_area}
```python
male_df = df[df.gender == 0].reset_index(drop=True)
female_df = df[df.gender == 1].reset_index(drop=True)
```



Boolean or logical indexing is useful if you need to sort the data based on some True or False value.  

For instance, who are the people with salaries greater than 90K but lower than 100K ?



{:.input_area}
```python
df[ (df.salary > 90000) & (df.salary < 100000)]
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>97630</td>
      <td>0</td>
      <td>chem</td>
      <td>34.0</td>
      <td>64.0</td>
      <td>43</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>30</th>
      <td>92951</td>
      <td>0</td>
      <td>neuro</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>20</td>
      <td>1.818182</td>
    </tr>
    <tr>
      <th>54</th>
      <td>96936</td>
      <td>0</td>
      <td>physics</td>
      <td>15.0</td>
      <td>50.0</td>
      <td>17</td>
      <td>1.133333</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Dealing with missing values
It is easy to quickly count the number of missing values for each column in the dataset using the `isnull()` method.  One thing that is  nice about Python is that you can chain commands, which means that the output of one method can be the input into the next method.  This allows us to write intuitive and concise code.  Notice how we take the `sum()` of all of the null cases.

The `isnull()` method will return a dataframe with True/False values on whether a datapoint is null or not a number (nan).



{:.input_area}
```python
df.isnull()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>51</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>52</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>53</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>54</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>55</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>56</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>57</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>58</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>59</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>60</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>61</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>62</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>63</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>64</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>65</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>66</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>67</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>68</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>69</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>70</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>71</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>72</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>73</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>74</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>75</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>76</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 7 columns</p>
</div>
</div>



We can chain the `.null()` and `.sum()` methods to see how many null values are added up.




{:.input_area}
```python
df.isnull().sum()
```





{:.output .output_data_text}
```
salary          0
gender          0
departm         0
years           1
age             1
publications    0
pubperyear      1
dtype: int64
```



You can use the boolean indexing once again to see the datapoints that have missing values. We chained the method `.any()` which will check if there are any True values for a given axis.  Axis=0 indicates rows, while Axis=1 indicates columns.  So here we are creating a boolean index for row where *any* column has a missing value.



{:.input_area}
```python
df[df.isnull().any(axis=1)]
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>64762</td>
      <td>0</td>
      <td>chem</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>24</th>
      <td>104828</td>
      <td>0</td>
      <td>geol</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>44</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</div>



You may look at where the values are not null. Note that indexes 18, and 24 are missing. 



{:.input_area}
```python
df[~df.isnull().any(axis=1)]
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
      <th>salary_in_departm</th>
      <th>dept_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
      <td>2.769231</td>
      <td>2.468065</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
      <td>1.535714</td>
      <td>1.493198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.300000</td>
      <td>0.939461</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
      <td>4.000000</td>
      <td>0.788016</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
      <td>2.090909</td>
      <td>0.375613</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64451</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>44</td>
      <td>1.913043</td>
      <td>0.144348</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>64366</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>22</td>
      <td>0.956522</td>
      <td>0.135301</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59344</td>
      <td>0</td>
      <td>bio</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>11</td>
      <td>2.200000</td>
      <td>-0.399173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>58560</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>8</td>
      <td>1.000000</td>
      <td>-0.482611</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>58294</td>
      <td>0</td>
      <td>bio</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>12</td>
      <td>0.600000</td>
      <td>-0.510921</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>56092</td>
      <td>0</td>
      <td>bio</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>4</td>
      <td>2.000000</td>
      <td>-0.745272</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54452</td>
      <td>0</td>
      <td>bio</td>
      <td>13.0</td>
      <td>43.0</td>
      <td>7</td>
      <td>0.538462</td>
      <td>-0.919812</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54269</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>12</td>
      <td>0.461538</td>
      <td>-0.939288</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>55125</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>9</td>
      <td>1.125000</td>
      <td>-0.848186</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>97630</td>
      <td>0</td>
      <td>chem</td>
      <td>34.0</td>
      <td>64.0</td>
      <td>43</td>
      <td>1.264706</td>
      <td>1.900129</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>82444</td>
      <td>0</td>
      <td>chem</td>
      <td>31.0</td>
      <td>61.0</td>
      <td>42</td>
      <td>1.354839</td>
      <td>0.984156</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>76291</td>
      <td>0</td>
      <td>chem</td>
      <td>29.0</td>
      <td>65.0</td>
      <td>33</td>
      <td>1.137931</td>
      <td>0.613025</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>75382</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>39</td>
      <td>1.500000</td>
      <td>0.558197</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62607</td>
      <td>0</td>
      <td>chem</td>
      <td>20.0</td>
      <td>45.0</td>
      <td>34</td>
      <td>1.700000</td>
      <td>-0.212352</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60373</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>43</td>
      <td>1.653846</td>
      <td>-0.347100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>58892</td>
      <td>0</td>
      <td>chem</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
      <td>1.166667</td>
      <td>-0.436429</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>47021</td>
      <td>0</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
      <td>3.000000</td>
      <td>-1.152452</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>44687</td>
      <td>0</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>19</td>
      <td>4.750000</td>
      <td>-1.293232</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>71456</td>
      <td>0</td>
      <td>geol</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>32</td>
      <td>2.909091</td>
      <td>0.876557</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>65144</td>
      <td>0</td>
      <td>geol</td>
      <td>7.0</td>
      <td>37.0</td>
      <td>12</td>
      <td>1.714286</td>
      <td>0.212671</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>52766</td>
      <td>0</td>
      <td>geol</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>32</td>
      <td>8.000000</td>
      <td>-1.089228</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>112800</td>
      <td>0</td>
      <td>neuro</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>33</td>
      <td>2.357143</td>
      <td>2.024705</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>105761</td>
      <td>0</td>
      <td>neuro</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>30</td>
      <td>3.333333</td>
      <td>1.632462</td>
      <td>3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>92951</td>
      <td>0</td>
      <td>neuro</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>20</td>
      <td>1.818182</td>
      <td>0.918635</td>
      <td>3</td>
    </tr>
    <tr>
      <th>31</th>
      <td>86621</td>
      <td>0</td>
      <td>neuro</td>
      <td>19.0</td>
      <td>49.0</td>
      <td>10</td>
      <td>0.526316</td>
      <td>0.565901</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>69596</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>18</td>
      <td>0.900000</td>
      <td>0.158420</td>
      <td>4</td>
    </tr>
    <tr>
      <th>47</th>
      <td>65285</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>15</td>
      <td>0.750000</td>
      <td>-0.131801</td>
      <td>4</td>
    </tr>
    <tr>
      <th>48</th>
      <td>62557</td>
      <td>0</td>
      <td>stat</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>14</td>
      <td>0.500000</td>
      <td>-0.315453</td>
      <td>4</td>
    </tr>
    <tr>
      <th>49</th>
      <td>61947</td>
      <td>0</td>
      <td>stat</td>
      <td>22.0</td>
      <td>58.0</td>
      <td>17</td>
      <td>0.772727</td>
      <td>-0.356519</td>
      <td>4</td>
    </tr>
    <tr>
      <th>50</th>
      <td>58565</td>
      <td>0</td>
      <td>stat</td>
      <td>29.0</td>
      <td>59.0</td>
      <td>11</td>
      <td>0.379310</td>
      <td>-0.584199</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51</th>
      <td>58365</td>
      <td>0</td>
      <td>stat</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
      <td>1.166667</td>
      <td>-0.597664</td>
      <td>4</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53656</td>
      <td>0</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>4</td>
      <td>2.000000</td>
      <td>-0.914679</td>
      <td>4</td>
    </tr>
    <tr>
      <th>53</th>
      <td>51391</td>
      <td>0</td>
      <td>stat</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>8</td>
      <td>1.600000</td>
      <td>-1.067161</td>
      <td>4</td>
    </tr>
    <tr>
      <th>54</th>
      <td>96936</td>
      <td>0</td>
      <td>physics</td>
      <td>15.0</td>
      <td>50.0</td>
      <td>17</td>
      <td>1.133333</td>
      <td>1.909602</td>
      <td>5</td>
    </tr>
    <tr>
      <th>55</th>
      <td>83216</td>
      <td>0</td>
      <td>physics</td>
      <td>11.0</td>
      <td>37.0</td>
      <td>19</td>
      <td>1.727273</td>
      <td>1.004571</td>
      <td>5</td>
    </tr>
    <tr>
      <th>56</th>
      <td>72044</td>
      <td>0</td>
      <td>physics</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>16</td>
      <td>8.000000</td>
      <td>0.267617</td>
      <td>5</td>
    </tr>
    <tr>
      <th>57</th>
      <td>64048</td>
      <td>0</td>
      <td>physics</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>4</td>
      <td>0.173913</td>
      <td>-0.259834</td>
      <td>5</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58888</td>
      <td>0</td>
      <td>physics</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>7</td>
      <td>0.269231</td>
      <td>-0.600210</td>
      <td>5</td>
    </tr>
    <tr>
      <th>59</th>
      <td>58744</td>
      <td>0</td>
      <td>physics</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>9</td>
      <td>0.450000</td>
      <td>-0.609708</td>
      <td>5</td>
    </tr>
    <tr>
      <th>60</th>
      <td>55944</td>
      <td>0</td>
      <td>physics</td>
      <td>21.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>0.380952</td>
      <td>-0.794409</td>
      <td>5</td>
    </tr>
    <tr>
      <th>61</th>
      <td>54076</td>
      <td>0</td>
      <td>physics</td>
      <td>19.0</td>
      <td>49.0</td>
      <td>12</td>
      <td>0.631579</td>
      <td>-0.917630</td>
      <td>5</td>
    </tr>
    <tr>
      <th>62</th>
      <td>82142</td>
      <td>0</td>
      <td>math</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>9</td>
      <td>1.000000</td>
      <td>1.810331</td>
      <td>6</td>
    </tr>
    <tr>
      <th>63</th>
      <td>70509</td>
      <td>0</td>
      <td>math</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>7</td>
      <td>0.304348</td>
      <td>0.765887</td>
      <td>6</td>
    </tr>
    <tr>
      <th>64</th>
      <td>60320</td>
      <td>0</td>
      <td>math</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>7</td>
      <td>0.500000</td>
      <td>-0.148911</td>
      <td>6</td>
    </tr>
    <tr>
      <th>65</th>
      <td>55814</td>
      <td>0</td>
      <td>math</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>6</td>
      <td>0.750000</td>
      <td>-0.553473</td>
      <td>6</td>
    </tr>
    <tr>
      <th>66</th>
      <td>53638</td>
      <td>0</td>
      <td>math</td>
      <td>4.0</td>
      <td>42.0</td>
      <td>8</td>
      <td>2.000000</td>
      <td>-0.748841</td>
      <td>6</td>
    </tr>
    <tr>
      <th>68</th>
      <td>59139</td>
      <td>1</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.875000</td>
      <td>-0.420990</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>52968</td>
      <td>1</td>
      <td>bio</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>32</td>
      <td>1.777778</td>
      <td>-1.077749</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>55949</td>
      <td>1</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
      <td>3.000000</td>
      <td>-0.613942</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>58893</td>
      <td>1</td>
      <td>neuro</td>
      <td>10.0</td>
      <td>35.0</td>
      <td>4</td>
      <td>0.400000</td>
      <td>-0.979219</td>
      <td>3</td>
    </tr>
    <tr>
      <th>72</th>
      <td>53662</td>
      <td>1</td>
      <td>neuro</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>3</td>
      <td>3.000000</td>
      <td>-1.270712</td>
      <td>3</td>
    </tr>
    <tr>
      <th>73</th>
      <td>57185</td>
      <td>1</td>
      <td>stat</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>7</td>
      <td>0.777778</td>
      <td>-0.677103</td>
      <td>4</td>
    </tr>
    <tr>
      <th>74</th>
      <td>52254</td>
      <td>1</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>9</td>
      <td>4.500000</td>
      <td>-1.009063</td>
      <td>4</td>
    </tr>
    <tr>
      <th>75</th>
      <td>61885</td>
      <td>1</td>
      <td>math</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>9</td>
      <td>0.391304</td>
      <td>-0.008401</td>
      <td>6</td>
    </tr>
    <tr>
      <th>76</th>
      <td>49542</td>
      <td>1</td>
      <td>math</td>
      <td>3.0</td>
      <td>33.0</td>
      <td>5</td>
      <td>1.666667</td>
      <td>-1.116592</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 9 columns</p>
</div>
</div>



There are different techniques for dealing with missing data.  An easy one is to simply remove rows that have any missing values using the `dropna()` method.



{:.input_area}
```python
df = df.dropna()
```


Now we can check to make sure the missing rows are removed.  Let's also check the new dimensions of the dataframe.



{:.input_area}
```python
print('There are %i rows and %i columns in this data set' % df.shape)
df.isnull().sum()
```


{:.output .output_stream}
```
There are 75 rows and 7 columns in this data set

```




{:.output .output_data_text}
```
salary          0
gender          0
departm         0
years           0
age             0
publications    0
pubperyear      0
dtype: int64
```



## Describing the data
We can use the `.describe()` method to get a quick summary of the continuous values of the data frame. We will `.transpose()` the output to make it slightly easier to read. 



{:.input_area}
```python
df.describe().transpose()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>salary</th>
      <td>75.0</td>
      <td>67293.946667</td>
      <td>14672.455177</td>
      <td>44687.000000</td>
      <td>56638.500000</td>
      <td>62557.000000</td>
      <td>74733.500000</td>
      <td>112800.0</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>75.0</td>
      <td>0.146667</td>
      <td>0.392268</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>years</th>
      <td>75.0</td>
      <td>14.840000</td>
      <td>8.596102</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>14.000000</td>
      <td>22.500000</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>75.0</td>
      <td>45.426667</td>
      <td>9.051166</td>
      <td>31.000000</td>
      <td>38.000000</td>
      <td>44.000000</td>
      <td>53.000000</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>publications</th>
      <td>75.0</td>
      <td>21.440000</td>
      <td>15.200676</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>18.000000</td>
      <td>32.500000</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>pubperyear</th>
      <td>75.0</td>
      <td>1.926595</td>
      <td>1.602968</td>
      <td>0.173913</td>
      <td>0.775253</td>
      <td>1.653846</td>
      <td>2.563187</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



We can also get quick summary of a pandas series, or specific column of a pandas dataframe.



{:.input_area}
```python
df.departm.describe()
```





{:.output .output_data_text}
```
count      75
unique      7
top       bio
freq       16
Name: departm, dtype: object
```



## Manipulating data in Groups
One manipulation we often do is look at variables in groups. 
One way to do this is to usethe `.groupby(key)` method. 
The key is a column that is used to group the variables together. 
For instance, if we want to group the data by gender and get group means, we perform the following.



{:.input_area}
```python
df.groupby('gender').mean()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69108.492308</td>
      <td>15.846154</td>
      <td>46.492308</td>
      <td>23.061538</td>
      <td>1.924709</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55719.666667</td>
      <td>8.666667</td>
      <td>38.888889</td>
      <td>11.555556</td>
      <td>2.043170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53517.000000</td>
      <td>5.000000</td>
      <td>35.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Other default aggregation methods include .count(), .mean(), .median(), .min(), .max(), .std(), .var(), and .sum()

Before we move on, it looks like there were more than 2 genders specified in our data. 
This is likely an error in the data collection process so let recap on how we might remove this datapoint. 



{:.input_area}
```python
df[df['gender']==2]
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>53517</td>
      <td>2</td>
      <td>math</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>



replace original dataframe without the miscoded data




{:.input_area}
```python
df = df[df['gender']!=2]
```


Now we have a corrected dataframe!




{:.input_area}
```python
df.groupby('gender').mean()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69108.492308</td>
      <td>15.846154</td>
      <td>46.492308</td>
      <td>23.061538</td>
      <td>1.924709</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55719.666667</td>
      <td>8.666667</td>
      <td>38.888889</td>
      <td>11.555556</td>
      <td>2.043170</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Another powerful tool in Pandas is the split-apply-combine method. 
For instance, let's say we also want to look at how much each professor is earning in respect to the department. 
Let's say we want to subtract the departmental mean from professor and divide it by the departmental standard deviation. 
We can do this by using the `groupby(key)` method chained with the `.transform(function)` method. 
It will group the dataframe by the key column, perform the "function" transformation of the data and return data in same format.
To learn more, see link [here](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.08-Aggregation-and-Grouping.ipynb)



{:.input_area}
```python
# key: We use the departm as the grouping factor. 
key = df['departm']

# Let's create an anonmyous function for calculating zscores using lambda:
# We want to standardize salary for each department.
zscore = lambda x: (x - x.mean()) / x.std()

# Now let's calculate zscores separately within each department
transformed = df.groupby(key).transform(zscore)
df['salary_in_departm'] = transformed['salary']
```


Now we have `salary_in_departm` column showing standardized salary per department.




{:.input_area}
```python
df.head()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
      <th>salary_in_departm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
      <td>2.769231</td>
      <td>2.468065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
      <td>1.535714</td>
      <td>1.493198</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.300000</td>
      <td>0.939461</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
      <td>4.000000</td>
      <td>0.788016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
      <td>2.090909</td>
      <td>0.375613</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Combining datasets - pd.concat
Recall that we sliced the dataframes into male and female dataframe in 2.3 Indexing and Slicing. Now we will learn how to put dataframes together which is done by the pd.concat method. Note how the index of this output retains the old index.



{:.input_area}
```python
pd.concat([femaledf, maledf],axis = 0)
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59139</td>
      <td>1</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.875000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52968</td>
      <td>1</td>
      <td>bio</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>32</td>
      <td>1.777778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55949</td>
      <td>1</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58893</td>
      <td>1</td>
      <td>neuro</td>
      <td>10.0</td>
      <td>35.0</td>
      <td>4</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53662</td>
      <td>1</td>
      <td>neuro</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>3</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>57185</td>
      <td>1</td>
      <td>stat</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>7</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>6</th>
      <td>52254</td>
      <td>1</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>9</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>61885</td>
      <td>1</td>
      <td>math</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>9</td>
      <td>0.391304</td>
    </tr>
    <tr>
      <th>8</th>
      <td>49542</td>
      <td>1</td>
      <td>math</td>
      <td>3.0</td>
      <td>33.0</td>
      <td>5</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
      <td>2.769231</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
      <td>1.535714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
      <td>2.090909</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64451</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>44</td>
      <td>1.913043</td>
    </tr>
    <tr>
      <th>6</th>
      <td>64366</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>22</td>
      <td>0.956522</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59344</td>
      <td>0</td>
      <td>bio</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>11</td>
      <td>2.200000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>58560</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>8</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>58294</td>
      <td>0</td>
      <td>bio</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>12</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>56092</td>
      <td>0</td>
      <td>bio</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>4</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54452</td>
      <td>0</td>
      <td>bio</td>
      <td>13.0</td>
      <td>43.0</td>
      <td>7</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54269</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>12</td>
      <td>0.461538</td>
    </tr>
    <tr>
      <th>13</th>
      <td>55125</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>9</td>
      <td>1.125000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>97630</td>
      <td>0</td>
      <td>chem</td>
      <td>34.0</td>
      <td>64.0</td>
      <td>43</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>15</th>
      <td>82444</td>
      <td>0</td>
      <td>chem</td>
      <td>31.0</td>
      <td>61.0</td>
      <td>42</td>
      <td>1.354839</td>
    </tr>
    <tr>
      <th>16</th>
      <td>76291</td>
      <td>0</td>
      <td>chem</td>
      <td>29.0</td>
      <td>65.0</td>
      <td>33</td>
      <td>1.137931</td>
    </tr>
    <tr>
      <th>17</th>
      <td>75382</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>39</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>64762</td>
      <td>0</td>
      <td>chem</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>1.160000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62607</td>
      <td>0</td>
      <td>chem</td>
      <td>20.0</td>
      <td>45.0</td>
      <td>34</td>
      <td>1.700000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60373</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>43</td>
      <td>1.653846</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>66482</td>
      <td>0</td>
      <td>neuro</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>42</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>61680</td>
      <td>0</td>
      <td>neuro</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>20</td>
      <td>1.111111</td>
    </tr>
    <tr>
      <th>39</th>
      <td>60455</td>
      <td>0</td>
      <td>neuro</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>49</td>
      <td>6.125000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>58932</td>
      <td>0</td>
      <td>neuro</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>49</td>
      <td>4.454545</td>
    </tr>
    <tr>
      <th>41</th>
      <td>106412</td>
      <td>0</td>
      <td>stat</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>29</td>
      <td>1.260870</td>
    </tr>
    <tr>
      <th>42</th>
      <td>86980</td>
      <td>0</td>
      <td>stat</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>42</td>
      <td>1.826087</td>
    </tr>
    <tr>
      <th>43</th>
      <td>78114</td>
      <td>0</td>
      <td>stat</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>24</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>74085</td>
      <td>0</td>
      <td>stat</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>33</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>72250</td>
      <td>0</td>
      <td>stat</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>9</td>
      <td>0.346154</td>
    </tr>
    <tr>
      <th>46</th>
      <td>69596</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>18</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>65285</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>15</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>62557</td>
      <td>0</td>
      <td>stat</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>14</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>61947</td>
      <td>0</td>
      <td>stat</td>
      <td>22.0</td>
      <td>58.0</td>
      <td>17</td>
      <td>0.772727</td>
    </tr>
    <tr>
      <th>50</th>
      <td>58565</td>
      <td>0</td>
      <td>stat</td>
      <td>29.0</td>
      <td>59.0</td>
      <td>11</td>
      <td>0.379310</td>
    </tr>
    <tr>
      <th>51</th>
      <td>58365</td>
      <td>0</td>
      <td>stat</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
      <td>1.166667</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53656</td>
      <td>0</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>4</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>51391</td>
      <td>0</td>
      <td>stat</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>8</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>96936</td>
      <td>0</td>
      <td>physics</td>
      <td>15.0</td>
      <td>50.0</td>
      <td>17</td>
      <td>1.133333</td>
    </tr>
    <tr>
      <th>55</th>
      <td>83216</td>
      <td>0</td>
      <td>physics</td>
      <td>11.0</td>
      <td>37.0</td>
      <td>19</td>
      <td>1.727273</td>
    </tr>
    <tr>
      <th>56</th>
      <td>72044</td>
      <td>0</td>
      <td>physics</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>16</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>64048</td>
      <td>0</td>
      <td>physics</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>4</td>
      <td>0.173913</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58888</td>
      <td>0</td>
      <td>physics</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>7</td>
      <td>0.269231</td>
    </tr>
    <tr>
      <th>59</th>
      <td>58744</td>
      <td>0</td>
      <td>physics</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>9</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>60</th>
      <td>55944</td>
      <td>0</td>
      <td>physics</td>
      <td>21.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>0.380952</td>
    </tr>
    <tr>
      <th>61</th>
      <td>54076</td>
      <td>0</td>
      <td>physics</td>
      <td>19.0</td>
      <td>49.0</td>
      <td>12</td>
      <td>0.631579</td>
    </tr>
    <tr>
      <th>62</th>
      <td>82142</td>
      <td>0</td>
      <td>math</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>9</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>63</th>
      <td>70509</td>
      <td>0</td>
      <td>math</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>7</td>
      <td>0.304348</td>
    </tr>
    <tr>
      <th>64</th>
      <td>60320</td>
      <td>0</td>
      <td>math</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>7</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>65</th>
      <td>55814</td>
      <td>0</td>
      <td>math</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>6</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>53638</td>
      <td>0</td>
      <td>math</td>
      <td>4.0</td>
      <td>42.0</td>
      <td>8</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 7 columns</p>
</div>
</div>



We can reset the index to start at zero using the .reset_index() method



{:.input_area}
```python
pd.concat([maledf, femaledf], axis = 0).reset_index(drop=True)
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86285</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>64.0</td>
      <td>72</td>
      <td>2.769231</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77125</td>
      <td>0</td>
      <td>bio</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>43</td>
      <td>1.535714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71922</td>
      <td>0</td>
      <td>bio</td>
      <td>10.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70499</td>
      <td>0</td>
      <td>bio</td>
      <td>16.0</td>
      <td>46.0</td>
      <td>64</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66624</td>
      <td>0</td>
      <td>bio</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>23</td>
      <td>2.090909</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64451</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>44</td>
      <td>1.913043</td>
    </tr>
    <tr>
      <th>6</th>
      <td>64366</td>
      <td>0</td>
      <td>bio</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>22</td>
      <td>0.956522</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59344</td>
      <td>0</td>
      <td>bio</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>11</td>
      <td>2.200000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>58560</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>8</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>58294</td>
      <td>0</td>
      <td>bio</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>12</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>56092</td>
      <td>0</td>
      <td>bio</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>4</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54452</td>
      <td>0</td>
      <td>bio</td>
      <td>13.0</td>
      <td>43.0</td>
      <td>7</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54269</td>
      <td>0</td>
      <td>bio</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>12</td>
      <td>0.461538</td>
    </tr>
    <tr>
      <th>13</th>
      <td>55125</td>
      <td>0</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>9</td>
      <td>1.125000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>97630</td>
      <td>0</td>
      <td>chem</td>
      <td>34.0</td>
      <td>64.0</td>
      <td>43</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>15</th>
      <td>82444</td>
      <td>0</td>
      <td>chem</td>
      <td>31.0</td>
      <td>61.0</td>
      <td>42</td>
      <td>1.354839</td>
    </tr>
    <tr>
      <th>16</th>
      <td>76291</td>
      <td>0</td>
      <td>chem</td>
      <td>29.0</td>
      <td>65.0</td>
      <td>33</td>
      <td>1.137931</td>
    </tr>
    <tr>
      <th>17</th>
      <td>75382</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>39</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>64762</td>
      <td>0</td>
      <td>chem</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>29</td>
      <td>1.160000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62607</td>
      <td>0</td>
      <td>chem</td>
      <td>20.0</td>
      <td>45.0</td>
      <td>34</td>
      <td>1.700000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60373</td>
      <td>0</td>
      <td>chem</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>43</td>
      <td>1.653846</td>
    </tr>
    <tr>
      <th>21</th>
      <td>58892</td>
      <td>0</td>
      <td>chem</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
      <td>1.166667</td>
    </tr>
    <tr>
      <th>22</th>
      <td>47021</td>
      <td>0</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>44687</td>
      <td>0</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>19</td>
      <td>4.750000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>104828</td>
      <td>0</td>
      <td>geol</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>44</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>71456</td>
      <td>0</td>
      <td>geol</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>32</td>
      <td>2.909091</td>
    </tr>
    <tr>
      <th>26</th>
      <td>65144</td>
      <td>0</td>
      <td>geol</td>
      <td>7.0</td>
      <td>37.0</td>
      <td>12</td>
      <td>1.714286</td>
    </tr>
    <tr>
      <th>27</th>
      <td>52766</td>
      <td>0</td>
      <td>geol</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>32</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>112800</td>
      <td>0</td>
      <td>neuro</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>33</td>
      <td>2.357143</td>
    </tr>
    <tr>
      <th>29</th>
      <td>105761</td>
      <td>0</td>
      <td>neuro</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>30</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>69596</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>18</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>65285</td>
      <td>0</td>
      <td>stat</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>15</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>62557</td>
      <td>0</td>
      <td>stat</td>
      <td>28.0</td>
      <td>58.0</td>
      <td>14</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>61947</td>
      <td>0</td>
      <td>stat</td>
      <td>22.0</td>
      <td>58.0</td>
      <td>17</td>
      <td>0.772727</td>
    </tr>
    <tr>
      <th>50</th>
      <td>58565</td>
      <td>0</td>
      <td>stat</td>
      <td>29.0</td>
      <td>59.0</td>
      <td>11</td>
      <td>0.379310</td>
    </tr>
    <tr>
      <th>51</th>
      <td>58365</td>
      <td>0</td>
      <td>stat</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>21</td>
      <td>1.166667</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53656</td>
      <td>0</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>4</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>51391</td>
      <td>0</td>
      <td>stat</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>8</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>96936</td>
      <td>0</td>
      <td>physics</td>
      <td>15.0</td>
      <td>50.0</td>
      <td>17</td>
      <td>1.133333</td>
    </tr>
    <tr>
      <th>55</th>
      <td>83216</td>
      <td>0</td>
      <td>physics</td>
      <td>11.0</td>
      <td>37.0</td>
      <td>19</td>
      <td>1.727273</td>
    </tr>
    <tr>
      <th>56</th>
      <td>72044</td>
      <td>0</td>
      <td>physics</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>16</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>64048</td>
      <td>0</td>
      <td>physics</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>4</td>
      <td>0.173913</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58888</td>
      <td>0</td>
      <td>physics</td>
      <td>26.0</td>
      <td>56.0</td>
      <td>7</td>
      <td>0.269231</td>
    </tr>
    <tr>
      <th>59</th>
      <td>58744</td>
      <td>0</td>
      <td>physics</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>9</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>60</th>
      <td>55944</td>
      <td>0</td>
      <td>physics</td>
      <td>21.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>0.380952</td>
    </tr>
    <tr>
      <th>61</th>
      <td>54076</td>
      <td>0</td>
      <td>physics</td>
      <td>19.0</td>
      <td>49.0</td>
      <td>12</td>
      <td>0.631579</td>
    </tr>
    <tr>
      <th>62</th>
      <td>82142</td>
      <td>0</td>
      <td>math</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>9</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>63</th>
      <td>70509</td>
      <td>0</td>
      <td>math</td>
      <td>23.0</td>
      <td>53.0</td>
      <td>7</td>
      <td>0.304348</td>
    </tr>
    <tr>
      <th>64</th>
      <td>60320</td>
      <td>0</td>
      <td>math</td>
      <td>14.0</td>
      <td>44.0</td>
      <td>7</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>65</th>
      <td>55814</td>
      <td>0</td>
      <td>math</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>6</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>53638</td>
      <td>0</td>
      <td>math</td>
      <td>4.0</td>
      <td>42.0</td>
      <td>8</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>67</th>
      <td>59139</td>
      <td>1</td>
      <td>bio</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>23</td>
      <td>2.875000</td>
    </tr>
    <tr>
      <th>68</th>
      <td>52968</td>
      <td>1</td>
      <td>bio</td>
      <td>18.0</td>
      <td>48.0</td>
      <td>32</td>
      <td>1.777778</td>
    </tr>
    <tr>
      <th>69</th>
      <td>55949</td>
      <td>1</td>
      <td>chem</td>
      <td>4.0</td>
      <td>34.0</td>
      <td>12</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>70</th>
      <td>58893</td>
      <td>1</td>
      <td>neuro</td>
      <td>10.0</td>
      <td>35.0</td>
      <td>4</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>71</th>
      <td>53662</td>
      <td>1</td>
      <td>neuro</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>3</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>72</th>
      <td>57185</td>
      <td>1</td>
      <td>stat</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>7</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>73</th>
      <td>52254</td>
      <td>1</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>9</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>74</th>
      <td>61885</td>
      <td>1</td>
      <td>math</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>9</td>
      <td>0.391304</td>
    </tr>
    <tr>
      <th>75</th>
      <td>49542</td>
      <td>1</td>
      <td>math</td>
      <td>3.0</td>
      <td>33.0</td>
      <td>5</td>
      <td>1.666667</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 7 columns</p>
</div>
</div>



## Plotting in pandas
Before we move into Matplotlib, here are a few plotting methods already implemented in Pandas. 
### Boxplot



{:.input_area}
```python
df[['salary','gender']].boxplot(by='gender')
```





{:.output .output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a1d97f320>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_67_1.png)



### Scatterplot



{:.input_area}
```python
df[['salary', 'years']].plot(kind='scatter', x='years', y='salary')
```





{:.output .output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a1dc9df28>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_69_1.png)



### Plotting Categorical Variables. Replacing variables with .map
If we want to plot department on the x-axis, Pandas plotting functions won't know what to do
because they don't know where to put bio or chem on a numerical x-axis. 
Therefore one needs to change them to numerical variable to plot them with basic functionalities (we will later see how Seaborn sovles this). 



{:.input_area}
```python
df['dept_num'] = 0
df.loc[:, ['dept_num']] = df.departm.map({'bio':0, 'chem':1, 'geol':2, 'neuro':3, 'stat':4, 'physics':5, 'math':6})
df.tail()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>gender</th>
      <th>departm</th>
      <th>years</th>
      <th>age</th>
      <th>publications</th>
      <th>pubperyear</th>
      <th>salary_in_departm</th>
      <th>dept_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>72</th>
      <td>53662</td>
      <td>1</td>
      <td>neuro</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>3</td>
      <td>3.000000</td>
      <td>-1.270712</td>
      <td>3</td>
    </tr>
    <tr>
      <th>73</th>
      <td>57185</td>
      <td>1</td>
      <td>stat</td>
      <td>9.0</td>
      <td>39.0</td>
      <td>7</td>
      <td>0.777778</td>
      <td>-0.677103</td>
      <td>4</td>
    </tr>
    <tr>
      <th>74</th>
      <td>52254</td>
      <td>1</td>
      <td>stat</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>9</td>
      <td>4.500000</td>
      <td>-1.009063</td>
      <td>4</td>
    </tr>
    <tr>
      <th>75</th>
      <td>61885</td>
      <td>1</td>
      <td>math</td>
      <td>23.0</td>
      <td>60.0</td>
      <td>9</td>
      <td>0.391304</td>
      <td>-0.008401</td>
      <td>6</td>
    </tr>
    <tr>
      <th>76</th>
      <td>49542</td>
      <td>1</td>
      <td>math</td>
      <td>3.0</td>
      <td>33.0</td>
      <td>5</td>
      <td>1.666667</td>
      <td>-1.116592</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Now plot all four categories



{:.input_area}
```python
f, axs = plt.subplots(1, 4, sharey=True)
f.suptitle('Salary in relation to other variables')
df.plot(kind='scatter', x='gender', y='salary', ax=axs[0], figsize=(15, 4))
df.plot(kind='scatter', x='dept_num', y='salary', ax=axs[1])
df.plot(kind='scatter', x='years', y='salary', ax=axs[2])
df.plot(kind='scatter', x='age', y='salary', ax=axs[3])
```





{:.output .output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a1dedb128>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_73_1.png)



The problem is that it treats department as a continuous variable. 

### Generating bar - errorbar plots in Pandas



{:.input_area}
```python
means = df.groupby('gender').mean()['salary']
errors = df.groupby('gender').std()['salary'] / np.sqrt(df.groupby('gender').count()['salary'])
ax = means.plot.bar(yerr=errors,figsize=(5,3))
```



{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_76_0.png)



# Matplotlib

Matplotlib is an object oriented plotting library in Python that is loosely based off of plotting in matlab.  It is the primary library that many other packages build on. Here is a very concise and helpful [introduction](https://github.com/neurohackademy/visualization-in-python/blob/master/visualization-in-python.ipynb) to plotting in Python.
Learn other matplotlib tutorials [here](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.00-Introduction-To-Matplotlib.ipynb)

## create a basic lineplot



{:.input_area}
```python
plt.figure(figsize=(8, 4))
plt.plot(range(0, 10),np.sqrt(range(0,10)), linewidth=3)
```





{:.output .output_data_text}
```
[<matplotlib.lines.Line2D at 0x1a1ee93630>]
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_78_1.png)



## create a basic scatterplot



{:.input_area}
```python
plt.figure(figsize=(4, 4))
plt.scatter(df.salary, df.age, color='b', marker='*')
```





{:.output .output_data_text}
```
<matplotlib.collections.PathCollection at 0x1a1ee3e2b0>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_80_1.png)



## Modify different aspects of the plot

`subplots` allows you to control different aspects of multiple plots



{:.input_area}
```python
f,ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4)) 
ax.scatter(df.salary, df.age,color='k', marker='o')
ax.set_xlim([40000,120000])
ax.set_ylim([20,70])
ax.set_xticklabels([str(int(tick)/1000)+'K' for tick in ax.get_xticks()])
ax.set_xlabel('Salary', fontsize=18)
ax.set_ylabel('Age', fontsize=18)
ax.set_title('Scatterplot of age and salary', fontsize=18)
```





{:.output .output_data_text}
```
Text(0.5, 1.0, 'Scatterplot of age and salary')
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_83_1.png)



We can save the plot using the `savefig` command.



{:.input_area}
```python
f.savefig('MyFirstPlot.png')
```


## Create multiple plots



{:.input_area}
```python
f,axs = plt.subplots(1, 2, figsize=(15,5)) 
axs[0].scatter(df.age, df.salary, color='k', marker='o')
axs[0].set_ylim([40000, 120000])
axs[0].set_xlim([20, 70])
axs[0].set_yticklabels([str(int(tick)/1000)+'K' for tick in axs[0].get_yticks()])
axs[0].set_ylabel('salary', fontsize=16)
axs[0].set_xlabel('age', fontsize=16)
axs[0].set_title('Scatterplot of age and salary', fontsize=16)

axs[1].scatter(df.publications, df.salary,color='k',marker='o')
axs[1].set_ylim([40000, 120000])
axs[1].set_xlim([20, 70])
axs[1].set_yticklabels([str(int(tick)/1000)+'K' for tick in axs[1].get_yticks()])

axs[1].set_ylabel('salary', fontsize=16)
axs[1].set_xlabel('publications', fontsize=16)
axs[1].set_title('Scatterplot of publication and salary', fontsize=16)

f.suptitle('Scatterplots of salary and other factors', fontsize=18)
```





{:.output .output_data_text}
```
Text(0.5, 0.98, 'Scatterplots of salary and other factors')
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_87_1.png)



# Seaborn
Seaborn is a plotting library built on Matplotlib that has many pre-configured plots that are often used for visualization. 
Other great tutorials about seaborn are [here](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.14-Visualization-With-Seaborn.ipynb)



{:.input_area}
```python
ax = sns.regplot(df.age, df.salary)
ax.set_title('Salary and age')
```





{:.output .output_data_text}
```
Text(0.5, 1.0, 'Salary and age')
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_89_1.png)





{:.input_area}
```python
sns.jointplot("age", "salary", data=df, kind='reg')
```





{:.output .output_data_text}
```
<seaborn.axisgrid.JointGrid at 0x1a1f0ebb00>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_90_1.png)



## Factor plots
Factor plots allow you to visualize the distribution of parameters in different forms such as point, bar, or violin graphs.  

Here are some possible values for kind : {point, bar, count, box, violin, strip}




{:.input_area}
```python
sns.catplot(x='departm', y='salary', hue='gender', data=df, ci=68, kind='bar')
```





{:.output .output_data_text}
```
<seaborn.axisgrid.FacetGrid at 0x1a1e49bc18>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_92_1.png)



## Heatmap plots
Heatmap plots allow you to visualize matrices such as correlation matrices that show relationships across multiple variables



{:.input_area}
```python
sns.heatmap(df[['salary', 'years', 'age', 'publications']].corr(), annot=True, linewidths=.5)
```





{:.output .output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a1fc82828>
```




{:.output .output_png}
![png](../../images/features/notebooks/2_Introduction_to_Dataframes_Plotting_94_1.png)



# Exercises ( Homework)
The following exercises uses the dataset "salary_exercise.csv" adapted from material available [here](http://data.princeton.edu/wws509/datasets/#salary)

These are the salary data used in Weisberg's book, consisting of observations on six variables for 52 tenure-track professors in a small college. The variables are:

 - sx = Sex, coded 1 for female and 0 for male
 - rk = Rank, coded
 - 1 for assistant professor,
 - 2 for associate professor, and
 - 3 for full professor
 - yr = Number of years in current rank
 - dg = Highest degree, coded 1 if doctorate, 0 if masters
 - yd = Number of years since highest degree was earned
 - sl = Academic year salary, in dollars.

Reference: S. Weisberg (1985). Applied Linear Regression, Second Edition. New York: John Wiley and Sons. Page 194.

## Exercise 1

Read the salary_exercise.csv into a dataframe, and change the column names to a more readable format such as 
sex, rank, yearsinrank, degree, yearssinceHD, and salary.   
Clean the data by excluding rows with any missing value. 
What are the overall mean, standard deviation, min, and maximum of professors' salary? 

## Exercise 2
Using the same data, what are the means and standard deviations of salary for different professor ranks?   
Create a new column on the original dataframe in which you calculate the standardized salary for each "rank" group. 

## Exercise 3
Recreate the plot shown in figure.   
On the left is a correlation of all parameters of only the male professors.  
On the right is the same but only for female professors.   
The colormap code used is `RdBu_r`. Read the Docstrings on sns.heatmap or search the internet to figure out how to change the colormap, scale the colorbar, and create square line boundaries.   
Place titles for each plot as shown, and your name as the main title.   

![](../images/labs/plotting/hw2-3.png)

## Exercise 4
Recreate the following plot from the salary_exercise.csv dataset.   
Create a 1 x 2 subplot.   
On the left is a bar-errorbar of salary per gender.   
On the right is a scatterplot of salary on y-axis and years in rank on the x-axis.  
Set the axis limits as shown in the picture and modify their lables.   
Add axis label names.   
Add a legend for the scatterplot and place it at a bottom-right location.  
Add your name as the main title of the plot.   

![](../images/labs/plottinghw2-4.png)
