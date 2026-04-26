import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    _ROOT = next(p for p in (Path.cwd(), *Path.cwd().resolve().parents) if (p / "book.yml").exists() or (p / "Code").is_dir())
    IMG_DIR = _ROOT / "images" / "pandas"
    return IMG_DIR, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Pandas

    *Written by Luke Chang & Jin Cheong*

    Analyzing data requires being facile with manipulating and transforming datasets to be able to test specific hypotheses. Data come in all different types of flavors and there are many different tools in the Python ecosystem to work with pretty much any type of data you might encounter. For example, you might be interested in working with functional neuroimaging data that is four dimensional. Three dimensional matrices contain brain activations in space, and these data can change over time in the 4th dimension. This type of data is well suited for [numpy](https://numpy.org/) and specialized brain imaging packages such as [nilearn](https://nilearn.github.io/). The majority of data, however, is typically in some version of a two-dimensional observations by features format as might be seen in an excel spreadsheet, a SQL table, or in a comma delimited format (i.e., csv).

    In Python, the [Pandas](https://pandas.pydata.org/) library is a powerful tool to work with this type of data. This is a very large library with a tremendous amount of functionality. In this tutorial, we will cover the basics of how to load and manipulate data and will focus on common to data munging tasks.

    For those interested in diving deeper into Pandas, there are many online resources. There is the [Pandas online documention](https://pandas.pydata.org/), [stackoverflow](https://stackoverflow.com/questions/tagged/pandas), and [medium blogposts](https://medium.com/search?q=pandas). I highly recommend  Jake Vanderplas's terrific [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/).   In addition, here is a brief [video](https://neurohackademy.org/course/complex-data-structures/) by Tal Yarkoni providing a useful introduction to pandas.

    After the tutorial you will have the chance to apply the methods to a new set of data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pandas Objects

    Pandas has several objects that are commonly used (i.e., Series, DataFrame, Index). At it's core, Pandas Objects are enhanced numpy arrays where columns and rows can have special names and there are lots of methods to operate on the data. See Jake Vanderplas's [tutorial](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.01-Introducing-Pandas-Objects.ipynb) for a more in depth overview.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Series
    A pandas `Series` is a one-dimensional array of indexed data.
    """)
    return


@app.cell
def _():
    import pandas as pd

    data = pd.Series([1, 2, 3, 4, 5])
    data
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The indices can be integers like in the example above. Alternatively, the indices can be labels.
    """)
    return


@app.cell
def _(pd):
    data_1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    data_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also, `Series` can be easily created from dictionaries
    """)
    return


@app.cell
def _(pd):
    data_2 = pd.Series({'A': 5, 'B': 3, 'C': 1})
    data_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DataFrame
    If a `Series` is a one-dimensional indexed array, the `DataFrame` is a two-dimensional indexed array. It can be thought of as a collection of Series objects, where each Series represents a column, or as an enhanced 2D numpy array.

    In a `DataFrame`, the index refers to labels for each row, while columns describe each column.

    First, let's create a `DataFrame` using random numbers generated from numpy.
    """)
    return


@app.cell
def _(pd):
    import numpy as np
    data_3 = pd.DataFrame(np.random.random((5, 3)))
    data_3
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could also initialize with column names
    """)
    return


@app.cell
def _(np, pd):
    data_4 = pd.DataFrame(np.random.random((5, 3)), columns=['A', 'B', 'C'])
    data_4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, we could create a `DataFrame` from multiple `Series` objects.
    """)
    return


@app.cell
def _(pd):
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series(['a', 'b', 'c', 'd'])
    data_5 = pd.DataFrame({'Numbers': a, 'Letters': b})
    data_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or a python dictionary
    """)
    return


@app.cell
def _(pd):
    data_6 = pd.DataFrame({'State': ['California', 'Colorado', 'New Hampshire'], 'Capital': ['Sacremento', 'Denver', 'Concord']})
    data_6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Data
    Loading data is fairly straightfoward in Pandas. Type `pd.read` then press tab to see a list of functions that can load specific file formats such as: csv, excel, spss, and sql.

    In this example, we will use `pd.read_csv` to load a .csv file into a dataframe.
    Note that read_csv() has many options that can be used to make sure you load the data correctly. You can explore the docstrings for a function to get more information about the inputs and general useage guidelines by running `pd.read_csv?`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To load a csv file we will need to specify either the relative or absolute path to the file.

    The command `pwd` will print the path of the current working directory.
    """)
    return


@app.cell
def _(pwd):
    pwd
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will now load the Pandas has many ways to read data different data formats into a dataframe.  Here we will use the `pd.read_csv` function.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('https://raw.githubusercontent.com/ljchang/dartbrains/master/data/salary/salary.csv', sep = ',')
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ways to check the dataframe
    There are many ways to examine your dataframe. One easy way is to just call the dataframe variable itself.
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, often the dataframes can be large and we may be only interested in seeing the first few rows.  `df.head()` is useful for this purpose.
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On the top row, you have column names, that can be called like a dictionary (a dataframe can be essentially thought of as a dictionary with column names as the keys). The left most column (0,1,2,3,4...) is called the index of the dataframe. The default index is sequential integers, but it can be set to anything as long as each row is unique (e.g., subject IDs)
    """)
    return


@app.cell
def _(df):
    print("Indexes")
    print(df.index)
    print("Columns")
    print(df.columns)
    print("Columns are like keys of a dictionary")
    print(df.keys())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can access the values of a column by calling it directly. Single bracket returns a `Series` and double bracket returns a `dataframe`.

    Let's return the first 10 rows of salary.
    """)
    return


@app.cell
def _(df):
    df['salary'][:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `shape` is another useful method for getting the dimensions of the matrix.

    We will print the number of rows and columns in this data set using fstring formatting. First, you need to specify a string starting with 'f', like this `f'anything'`. It is easy to insert variables with curly brackets like this `f'rows: {rows}'`.

    [Here](https://realpython.com/python-f-strings/) is more info about formatting text.
    """)
    return


@app.cell
def _(df):
    rows, cols = df.shape
    print(f'There are {rows} rows and {cols} columns in this data set')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Describing the data
    We can use the `.describe()` method to get a quick summary of the continuous values of the data frame. We will `.transpose()` the output to make it slightly easier to read.
    """)
    return


@app.cell
def _(df):
    df.describe().transpose()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also get quick summary of a pandas series, or specific column of a pandas dataframe.
    """)
    return


@app.cell
def _(df):
    df.departm.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sometimes, you will want to know how many data points are associated with a specific variable for categorical data. The `value_counts` method can be used for this goal.

    For example, how many males and females are in this dataset?
    """)
    return


@app.cell
def _(df):
    df['gender'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can see that there are more than 2 genders specified in our data.

    This is likely an error in the data collection process. It's always up to the data analyst to decide what to do in these cases. Because we don't know what the true value should have been, let's just remove the row from the dataframe by finding all rows that are not '2'.
    """)
    return


@app.cell
def _(df):
    df_1 = df.loc[df['gender'] != 2]
    df_1['gender'].value_counts()
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dealing with missing values
    Data are always messy and often have lots of missing values. There are many different ways, in which missing data might present `NaN`, `None`, or `NA`, Sometimes researchers code missing values with specific numeric codes such as 999999. It is important to find these as they can screw up your analyses if they are hiding in your data.

    If the missing values are using a standard pandas or numpy value such as `NaN`, `None`, or `NA`, we can identify where the missing values are as booleans using the `isnull()` method.

    The `isnull()` method will return a dataframe with True/False values on whether a datapoint is null or not a number (nan).
    """)
    return


@app.cell
def _(df_1):
    df_1.isnull()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Suppose we wanted to count the number of missing values for each column in the dataset.

    One thing that is  nice about Python is that you can chain commands, which means that the output of one method can be the input into the next method.  This allows us to write intuitive and concise code.  Notice how we take the `sum()` of all of the null cases.
    We can chain the `.null()` and `.sum()` methods to see how many null values are added up in each column.
    """)
    return


@app.cell
def _(df_1):
    df_1.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can use the boolean indexing once again to see the datapoints that have missing values. We chained the method `.any()` which will check if there are any True values for a given axis.  Axis=0 indicates rows, while Axis=1 indicates columns.  So here we are creating a boolean index for row where *any* column has a missing value.
    """)
    return


@app.cell
def _(df_1):
    df_1[df_1.isnull().any(axis=1)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You may look at where the values are not null. Note that indexes 18, and 24 are missing.
    """)
    return


@app.cell
def _(df_1):
    df_1[~df_1.isnull().any(axis=1)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are different techniques for dealing with missing data.  An easy one is to simply remove rows that have any missing values using the `dropna()` method.
    """)
    return


@app.cell
def _(df_1):
    df_1.dropna(inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can check to make sure the missing rows are removed.  Let's also check the new dimensions of the dataframe.
    """)
    return


@app.cell
def _(df_1):
    rows_1, cols_1 = df_1.shape
    print(f'There are {rows_1} rows and {cols_1} columns in this data set')
    df_1.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create New Columns
    You can create new columns to fit your needs.
    For instance you can set initialize a new column with zeros.
    """)
    return


@app.cell
def _(df_1):
    df_1['pubperyear'] = 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we can create a new column pubperyear, which is the ratio of the number of papers published per year
    """)
    return


@app.cell
def _(df_1):
    df_1['pubperyear'] = df_1['publications'] / df_1['years']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Indexing and slicing Data

    Indexing in Pandas can be tricky. There are many ways to index in pandas, for this tutorial we will focus on four: loc, iloc, boolean, and indexing numpy values. For a more in depth overview see Jake Vanderplas's tutorial](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.02-Data-Indexing-and-Selection.ipynb), where he also covers more advanced topics, such as [hierarchical indexing](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.05-Hierarchical-Indexing.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Indexing with Keys
    First, we will cover indexing with keys using the `.loc` method. This method references the explicit index with a key name. It works for both index names and also column names. Note that often the keys for rows are integers by default.

    In this example, we will return rows 10-20 on the salary column.
    """)
    return


@app.cell
def _(df_1):
    df_1.loc[10:20, 'salary']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can return multiple columns using a list.
    """)
    return


@app.cell
def _(df_1):
    df_1.loc[:10, ['salary', 'publications']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Indexing with Integers
    Next we wil try `.iloc`.  This method references the implicit python index using integer indexing (starting from 0, exclusive of last number).  You can think of this like row by column indexing using integers.

    For example, let's grab the first 3 rows and columns.
    """)
    return


@app.cell
def _(df_1):
    df_1.iloc[0:3, 0:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's make a new data frame with just Males and another for just Females. Notice, how we added the `.reset_index(drop=True)` method?   This is because assigning a new dataframe based on indexing another dataframe will retain the *original* index.  We need to explicitly tell pandas to reset the index if we want it to start from zero.
    """)
    return


@app.cell
def _(df_1):
    male_df = df_1[df_1.gender == 0].reset_index(drop=True)
    female_df = df_1[df_1.gender == 1].reset_index(drop=True)
    return female_df, male_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Indexing with booleans

    Boolean or logical indexing is useful if you need to sort the data based on some True or False value.

    For instance, who are the people with salaries greater than 90K but lower than 100K ?
    """)
    return


@app.cell
def _(df_1):
    df_1[(df_1.salary > 90000) & (df_1.salary < 100000)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This also works with the `.loc` method, which is what you need to do if you want to return specific columns
    """)
    return


@app.cell
def _(df_1):
    df_1.loc[(df_1.salary > 90000) & (df_1.salary < 100000), ['salary', 'gender']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numpy indexing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, you can also return a numpy matrix from a pandas data frame by accessing the `.values` property. This returns a numpy array that can be indexed using numpy integer indexing and slicing.

    As an example, let's grab the last 10 rows and the first 3 columns.
    """)
    return


@app.cell
def _(df_1):
    df_1.values[-10:, :3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Renaming
    Part of cleaning up the data is renaming with more sensible names. This is easy to do with Pandas.

    ### Renaming Columns
    We can rename columns with the `.rename` method by passing in a dictionary using the `{'Old Name':'New Name'}`. We either need to assigne the result to a new variable or add `inplace=True`.
    """)
    return


@app.cell
def _(df_1):
    df_1.rename({'departm': 'department', 'pubperyear': 'pub_per_year'}, axis=1, inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Renaming Rows
    Often we may want to change the coding scheme for a variable. For example, it is hard to remember what zeros and ones mean in the gender variable. We can make this easier by changing these with a dictionary `{0:'male', 1:'female'}` with the `replace` method. We can do this `inplace=True` or we can assign it to a new variable. As an example, we will assign this to a new variable to also retain the original lablels.
    """)
    return


@app.cell
def _(df_1):
    df_1['gender_name'] = df_1['gender'].replace({0: 'male', 1: 'female'})
    df_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Operations
    One of the really fun things about pandas once you get the hang of it is how easy it is to perform operations on the data. It is trivial to compute simple summaries of the data. We can also leverage the object-oriented nature of a pandas object, we can chain together multiple commands.

    For example, let's grab the mean of a few columns.
    """)
    return


@app.cell
def _(df_1):
    df_1.loc[:, ['years', 'age', 'publications']].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also turn these values into a plot with the `plot` method, which we will cover in more detail in future tutorials.
    """)
    return


@app.cell
def _(df_1):
    # '%matplotlib inline' command supported automatically in marimo
    df_1.loc[:, ['years', 'age', 'publications']].mean().plot(kind='bar')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Perhaps we want to see if there are any correlations in our dataset. We can do this with the `.corr` method. More recent versions of Pandas might produce an error if there are any columns containing string data. To avoid this issue set `numeric_only=True`.
    """)
    return


@app.cell
def _(df_1):
    df_1.corr(numeric_only=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Merging Data
    Another common data manipulation goal is to merge datasets. There are multiple ways to do this in pandas, we will cover concatenation, append, and merge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concatenation
    Concatenation describes the process of *stacking* dataframes together. Older versions of pandas also had an `.append()` method, which has been deprecated since pandas 1.4. The main thing to consider is to make sure that the shapes of the two dataframes are the same as well as the index labels. For example, if we wanted to vertically stack two dataframe, they need to have the same column names.

    Remember that we previously created two separate dataframes for males and females?  Let's put them back together using the `pd.concat` method. Note how the index of this output retains the old index.
    """)
    return


@app.cell
def _(female_df, male_df, pd):
    combined_data = pd.concat([female_df, male_df], axis = 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can reset the index using the `reset_index` method.
    """)
    return


@app.cell
def _(female_df, male_df, pd):
    pd.concat([male_df, female_df], axis = 0).reset_index(drop=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also concatenate columns in addition to rows. Make sure that the number of rows are the same in each dataframe. For this example, we will just create two new data frames with a subset of the columns and then combine them again.
    """)
    return


@app.cell
def _(df_1, pd):
    df1 = df_1[['salary', 'gender']]
    df2 = df_1[['age', 'publications']]
    df3 = pd.concat([df1, df2], axis=1)
    df3.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Merge
    The most powerful method of merging data is using the `pd.merge` method. This allows you to merge datasets of different shapes and sizes on specific variables that match. This is very common when you need to merge multiple sql tables together for example.

    In this example, we are creating two separate data frames that have different states and columns and will merge on the `State` column.

    First, we will only retain rows where there is a match across dataframes, using `how=inner`. This is equivalent to an 'and' join in sql.
    """)
    return


@app.cell
def _(pd):
    df1_1 = pd.DataFrame({'State': ['California', 'Colorado', 'New Hampshire'], 'Capital': ['Sacremento', 'Denver', 'Concord']})
    df2_1 = pd.DataFrame({'State': ['California', 'New Hampshire', 'New York'], 'Population': ['39512223', '1359711', '19453561']})
    df3_1 = pd.merge(left=df1_1, right=df2_1, on='State', how='inner')
    df3_1
    return df1_1, df2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how there are only two rows in the merged dataframe.

    We can also be more inclusive and match on `State` column, but retain all rows. This is equivalent to an 'or' join.
    """)
    return


@app.cell
def _(df1_1, df2_1, pd):
    df3_2 = pd.merge(left=df1_1, right=df2_1, on='State', how='outer')
    df3_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a very handy way to merge data when you have lots of files with missing data.  See Jake Vanderplas's [tutorial](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.07-Merge-and-Join.ipynb) for a more in depth overview.
    """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        ## Grouping
        We've seen above that it is very easy to summarize data over columns using the builtin functions such as `pd.mean()`. Sometimes we are interested in summarizing data over different groups of rows. For example, what is the mean of participants in Condition A compared to Condition B?

        This is suprisingly easy to compute in pandas using the `groupby` operator, where we aggregate data using a specific operation over different labels.

        One useful way to conceptualize this is using the **Split, Apply, Combine** operation (similar to map-reduce).
        """),
        mo.image(str(IMG_DIR / "split-apply-combine.png")),
        mo.md(r"""
        This figure is taken from Jake Vanderplas's tutorial and highlights how input data can be *split* on some key and then an operation such as sum can be *applied* separately to each split. Finally, the results of the applied function for each key can be *combined* into a new data frame.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Groupby
    In this example, we will use the `groupby` operator to split the data based on gender labels and separately calculate the mean for each group. Note that newer versions of pandas might throw an error if you try to perform a numeric computation such as `.mean()` on a dataframe containing columns of string data. Use the flag `numeric_only=True` to avoid this issue.
    """)
    return


@app.cell
def _(df_1):
    df_1.groupby('gender_name').mean(numeric_only=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other default aggregation methods include `.count()`, `.mean()`, `.median()`, `.min()`, `.max()`, `.std()`, `.var()`, and `.sum()`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transform
    While the split, apply, combine operation that we just demonstrated is extremely usefuly to quickly summarize data based on a grouping key, the resulting data frame is compressed to one row per grouping label.

    Sometimes, we would like to perform an operation over groups, but retain the original data shape. One common example is standardizing data within a subject or grouping variable. Normally, you might think to loop over subject ids and separately z-score or center a variable and then recombine the subject data using a vertical concatenation operation.

    The `transform` method in pandas can make this much easier and faster!

    Suppose we want to compute the standardized salary separately for each department. We can standardize using a z-score which requires subtracting the departmental mean from each professor's salary in that department, and then dividing it by the departmental standard deviation.

    We can do this by using the `groupby(key)` method chained with the `.transform(function)` method. It will group the dataframe by the key column, perform the "function" transformation of the data and return data in same format. We can then assign the results to a new column in the dataframe.
    """)
    return


@app.cell
def _(df_1):
    df_1['salary_dept_z'] = (df_1['salary'] - df_1['salary'].groupby(df_1['department']).transform('mean')) / df_1['salary'].groupby(df_1['department']).transform('std')
    df_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This worked well, but is also pretty verbose. We can simplify the syntax a little bit more using a `lambda` function, where we can define the zscore function.
    """)
    return


@app.cell
def _(df_1):
    calc_zscore = lambda x: (x - x.mean()) / x.std()
    df_1['salary_dept_z'] = df_1['salary'].groupby(df_1['department']).transform(calc_zscore)
    df_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a more in depth overview of data aggregation and grouping, check out Jake Vanderplas's [tutorial](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.08-Aggregation-and-Grouping.ipynb)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reshaping Data
    The last topic we will cover in this tutorial is reshaping data. Data is often in the form of observations by features, in which there is a single row for each independent observation of data and a separate column for each feature of the data. This is commonly referred to as as the *wide* format. However, when running regression or plotting in libraries such as seaborn, we often want our data in the *long* format, in which each grouping variable is specified in a separate column.

    In this section we cover how to go from wide to long using the `melt` operation and from long to wide using the `pivot` function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Melt
    To `melt` a dataframe into the long format, we need to specify which variables are the `id_vars`, which ones should be combined into a `value_var`, and finally, what we should label the column name for the `value_var`, and also for the `var_name`. We will call the values 'Ratings' and variables 'Condition'.

    First, we need to create a dataset to play with.
    """)
    return


@app.cell
def _(np, pd):
    data_7 = pd.DataFrame(np.vstack([np.arange(1, 6), np.random.random((3, 5))]).T, columns=['ID', 'A', 'B', 'C'])
    data_7
    return (data_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's melt the dataframe into the long format.
    """)
    return


@app.cell
def _(data_7, pd):
    df_long = pd.melt(data_7, id_vars='ID', value_vars=['A', 'B', 'C'], var_name='Condition', value_name='Rating')
    df_long
    return (df_long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the id variable is repeated for each condition?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pivot
    We can also go back to the wide data format from a long dataframe using `pivot`. We just need to specify the variable containing the labels which will become the `columns` and the `values` column that will be broken into separate columns.
    """)
    return


@app.cell
def _(df_long):
    df_wide = df_long.pivot(index='ID', columns='Condition', values='Rating')
    df_wide
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises ( Homework)
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

    ### Exercise 1

    Read the salary_exercise.csv into a dataframe, and change the column names to a more readable format such as sex, rank, yearsinrank, degree, yearssinceHD, and salary.

    Clean the data by excluding rows with any missing value.

    What are the overall mean, standard deviation, min, and maximum of professors' salary?
    """)
    return


@app.cell
def _():
    salary_file_url = 'https://raw.githubusercontent.com/ljchang/dartbrains/master/data/salary/salary_exercise.csv'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2
    Create two separate dataframes based on the type of degree. Now calculate the mean salary of the 5 oldest professors of each degree type.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3
    What is the correlation between the standardized salary *across* all ranks and the standardized salary *within* ranks?
    """)
    return


if __name__ == "__main__":
    app.run()
