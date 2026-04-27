import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", app_title="Introduction to Polars")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Polars

    *Written by Luke Chang*

    Polars is a blazing-fast DataFrame library for Python, written in Rust. It is designed for high-performance data manipulation and offers an expressive, consistent API that makes data wrangling both efficient and enjoyable. Unlike older tools, Polars was built from the ground up to take advantage of modern hardware through multi-threaded execution, lazy evaluation, and memory-efficient columnar storage.

    **Why Polars?**

    - **Speed**: Polars is one of the fastest DataFrame libraries available, routinely outperforming alternatives by 10-100x on large datasets.
    - **Memory efficiency**: Its Apache Arrow-based columnar format minimizes memory usage and avoids unnecessary copies.
    - **Expressive API**: The expression system lets you write concise, readable queries that are easy to compose and optimize.
    - **Lazy evaluation**: Polars can build an optimized query plan before executing, enabling automatic optimizations like predicate pushdown and projection pruning.

    In this tutorial, we will learn the fundamentals of Polars using a faculty salary dataset. By the end, you will be comfortable loading data, transforming columns, filtering rows, grouping and aggregating, and using advanced features like window functions and lazy evaluation.

    For more details, check out the official [Polars documentation](https://docs.pola.rs/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np

    return np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polars Objects

    Polars provides two core data structures: **Series** and **DataFrame**. Understanding these is the first step to working with Polars effectively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Series

    A `Series` is a typed, one-dimensional array. Every element in a Series has the same data type, which is determined at creation time. You can think of it as a single column of data.
    """)
    return


@app.cell
def _(pl):
    # Create a Series from a list of integers
    ages = pl.Series("age", [25, 30, 35, 40, 45])
    ages
    return


@app.cell
def _(pl):
    # Create a Series with an explicit type
    scores = pl.Series("score", [88.5, 92.0, 76.3, 95.1], dtype=pl.Float32)
    scores
    return


@app.cell
def _(pl):
    # String Series
    names = pl.Series("name", ["Alice", "Bob", "Charlie", "Diana"])
    names
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DataFrame

    A `DataFrame` is a two-dimensional table composed of multiple named Series (columns). Each column has its own data type, and all columns share the same length.

    A key difference from some other tools: **Polars DataFrames have no row index**. Rows are identified by their position, and any row labeling must be done explicitly through a column. This keeps the API simple and avoids ambiguity.
    """)
    return


@app.cell
def _(pl):
    # Create a DataFrame from a dictionary
    df_example = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 40],
        "salary": [50000, 60000, 70000, 80000],
    })
    df_example
    return


@app.cell
def _(np, pl):
    # Create a DataFrame from numpy arrays
    rng = np.random.default_rng(42)
    df_from_numpy = pl.DataFrame({
        "x": rng.normal(0, 1, 5),
        "y": rng.normal(0, 1, 5),
    })
    df_from_numpy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Data

    Polars can read data from many formats including CSV, Parquet, JSON, and more. We will load a faculty salary dataset that contains information about salaries, departments, years of experience, and other attributes.

    The `pl.read_csv()` function reads a CSV file eagerly (loading everything into memory immediately). Polars can also read directly from URLs.
    """)
    return


@app.cell
def _(pl):
    url = "https://raw.githubusercontent.com/ljchang/dartbrains/main/data/salary/salary.csv"
    df = pl.read_csv(url)
    df
    return df, url


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For very large files, you can use `pl.scan_csv()` instead to create a **LazyFrame** that defers execution until you call `.collect()`. We will explore lazy evaluation in detail later in this tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting Data

    Before diving into analysis, it is important to understand the structure and contents of your data. Polars provides several handy methods for this.
    """)
    return


@app.cell
def _(df):
    # View the first few rows
    df.head()
    return


@app.cell
def _(df):
    # View the last few rows
    df.tail()
    return


@app.cell
def _(df):
    # Random sample of rows
    df.sample(5, seed=42)
    return


@app.cell
def _(df):
    # Summary statistics for all columns
    df.describe()
    return


@app.cell
def _(df):
    # Column names and their data types
    df.schema
    return


@app.cell
def _(df):
    # Shape of the DataFrame (rows, columns)
    df.shape
    return


@app.cell
def _(df):
    # List of column names
    df.columns
    return


@app.cell
def _(df):
    # Check for missing values in each column
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dealing with Missing Values

    Polars uses `null` to represent missing data. This is distinct from `NaN` (Not a Number), which is a valid floating-point value. The `null` type is consistent across all data types, whether numeric, string, or boolean.
    """)
    return


@app.cell
def _(df):
    # See how many nulls each column has
    df.null_count()
    return


@app.cell
def _(df):
    # Drop all rows that contain any null value
    df_no_nulls = df.drop_nulls()
    print(f"Original rows: {df.shape[0]}, After dropping nulls: {df_no_nulls.shape[0]}")
    df_no_nulls
    return


@app.cell
def _(df, pl):
    # Fill nulls with a specific strategy (e.g., fill numeric nulls with the column mean)
    df_filled = df.with_columns(
        pl.col("years").fill_null(strategy="mean"),
        pl.col("age").fill_null(strategy="mean"),
    )
    df_filled.null_count()
    return


@app.cell
def _(df, pl):
    # Fill nulls with a constant value
    df.with_columns(
        pl.col("years").fill_null(0),
    ).head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Expression API

    The expression API is the heart of Polars. Expressions describe computations on columns without immediately executing them. They are the building blocks you pass to methods like `select()`, `filter()`, `with_columns()`, and `group_by().agg()`.

    The most common starting point is `pl.col("column_name")`, which references a column. From there, you can chain operations to build up complex transformations.
    """)
    return


@app.cell
def _(df, pl):
    # Reference a single column and compute its mean
    df.select(pl.col("salary").mean())
    return


@app.cell
def _(df, pl):
    # Chain multiple operations: compute several summary statistics at once
    df.select(
        pl.col("salary").mean().alias("mean_salary"),
        pl.col("salary").median().alias("median_salary"),
        pl.col("salary").min().alias("min_salary"),
        pl.col("salary").max().alias("max_salary"),
        pl.col("salary").std().alias("std_salary"),
    )
    return


@app.cell
def _(df, pl):
    # Compute the mean of multiple columns at once
    df.select(pl.col("salary", "years", "age", "publications").mean())
    return


@app.cell
def _(df, pl):
    # Use pl.all() to compute the sum of every numeric column
    df.select(pl.all().sum())
    return


@app.cell
def _(df, pl):
    # Conditional expressions with when/then/otherwise
    df.select(
        pl.col("salary"),
        pl.col("departm"),
        pl.when(pl.col("salary") > 80000)
          .then(pl.lit("high"))
          .otherwise(pl.lit("standard"))
          .alias("salary_tier"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating New Columns

    The `with_columns()` method adds new columns or transforms existing ones. It always returns a **new** DataFrame, leaving the original unchanged. This immutability makes your code easier to reason about and debug.

    Use `.alias("name")` to give your computed column a name.
    """)
    return


@app.cell
def _(df, pl):
    # Create a column showing salary in thousands
    df_enhanced = df.with_columns(
        (pl.col("salary") / 1000).round(1).alias("salary_thousands"),
    )
    df_enhanced.head()
    return


@app.cell
def _(df, pl):
    # Create multiple new columns at once
    df_multi = df.with_columns(
        (pl.col("salary") > 80000).alias("high_salary"),
        (pl.col("salary") / pl.col("publications")).round(0).alias("salary_per_pub"),
        (pl.col("age") - pl.col("years")).alias("age_at_start"),
    )
    df_multi.head()
    return


@app.cell
def _(df):
    # The original DataFrame is unchanged (immutability)
    print("Original columns:", df.columns)
    df.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting and Filtering

    `select()` chooses which columns to keep, while `filter()` chooses which rows to keep. Both accept Polars expressions.
    """)
    return


@app.cell
def _(df):
    # Select specific columns by name
    df.select("salary", "departm", "gender")
    return


@app.cell
def _(df, pl):
    # Select using expressions (allows renaming/transforming inline)
    df.select(
        pl.col("departm"),
        (pl.col("salary") / 1000).alias("salary_k"),
    )
    return


@app.cell
def _(df, pl):
    # Filter rows where salary exceeds 80000
    df.filter(pl.col("salary") > 80000)
    return


@app.cell
def _(df, pl):
    # Combine multiple filter conditions with & (and) and | (or)
    df.filter(
        (pl.col("salary") > 70000) & (pl.col("departm") == "neuro")
    )
    return


@app.cell
def _(df, pl):
    # Filter using is_in to match against a set of values
    df.filter(pl.col("departm").is_in(["neuro", "stat", "bio"]))
    return


@app.cell
def _(df):
    # Get unique departments
    df.select("departm").unique()
    return


@app.cell
def _(df):
    # Sort by salary descending
    df.sort("salary", descending=True).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Renaming Columns

    Use `rename()` to change column names. Pass a dictionary mapping old names to new names.
    """)
    return


@app.cell
def _(df):
    df_renamed = df.rename({
        "departm": "department",
        "years": "years_experience",
    })
    df_renamed.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Operations

    Polars supports a wide range of operations on columns, including arithmetic, string manipulations, and type casting. Whenever possible, use native Polars expressions rather than custom Python functions for best performance.
    """)
    return


@app.cell
def _(df, pl):
    # Arithmetic: give everyone a 10% raise
    df.select(
        pl.col("departm"),
        pl.col("salary"),
        (pl.col("salary") * 1.10).round(0).cast(pl.Int64).alias("salary_with_raise"),
    )
    return


@app.cell
def _(df, pl):
    # String operations: convert department names to uppercase
    df.select(
        pl.col("departm").str.to_uppercase().alias("department_upper"),
        pl.col("salary"),
    ).head()
    return


@app.cell
def _(df, pl):
    # Cast a column to a different type
    df.select(
        pl.col("salary").cast(pl.Float64).alias("salary_float"),
        pl.col("gender").cast(pl.Utf8).alias("gender_str"),
    ).head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For truly custom logic that cannot be expressed with native Polars expressions, you can use `map_elements()` to apply a Python function element-wise. However, this is significantly slower than native expressions because it bypasses Polars' optimized execution engine.
    """)
    return


@app.cell
def _(df, pl):
    # map_elements example (slow — prefer native expressions when possible)
    df.select(
        pl.col("departm"),
        pl.col("salary").map_elements(
            lambda x: f"${x:,}", return_dtype=pl.Utf8
        ).alias("salary_formatted"),
    ).head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Joining Data

    Polars supports several types of joins for combining DataFrames. The syntax is straightforward: call `.join()` on the left DataFrame and pass the right DataFrame along with the join key and type.
    """)
    return


@app.cell
def _(pl):
    # Create two example DataFrames to demonstrate joins
    departments = pl.DataFrame({
        "dept_code": ["bio", "chem", "neuro", "stat", "physics"],
        "full_name": ["Biology", "Chemistry", "Neuroscience", "Statistics", "Physics"],
        "building": ["LSC", "Burke", "Moore", "Kemeny", "Wilder"],
    })

    budgets = pl.DataFrame({
        "dept_code": ["bio", "chem", "neuro", "math", "geol"],
        "annual_budget": [500000, 750000, 900000, 300000, 400000],
    })
    return budgets, departments


@app.cell
def _(budgets, departments):
    # Inner join: only keeps rows where the key exists in both DataFrames
    departments.join(budgets, on="dept_code", how="inner")
    return


@app.cell
def _(budgets, departments):
    # Left join: keeps all rows from the left DataFrame
    departments.join(budgets, on="dept_code", how="left")
    return


@app.cell
def _(budgets, departments):
    # Full outer join: keeps all rows from both DataFrames
    departments.join(budgets, on="dept_code", how="full", coalesce=True)
    return


@app.cell
def _(pl):
    # Vertical stacking (concatenating rows)
    df_a = pl.DataFrame({"name": ["Alice", "Bob"], "score": [90, 85]})
    df_b = pl.DataFrame({"name": ["Charlie", "Diana"], "score": [78, 92]})
    pl.concat([df_a, df_b])
    return


@app.cell
def _(pl):
    # Horizontal stacking (concatenating columns)
    df_left = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    df_right = pl.DataFrame({"salary": [50000, 60000], "dept": ["bio", "chem"]})
    pl.concat([df_left, df_right], how="horizontal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouping and Aggregation

    Grouping is one of the most powerful operations in data analysis. Polars uses `group_by()` to split a DataFrame by one or more columns, then `agg()` to compute summary statistics within each group.
    """)
    return


@app.cell
def _(df, pl):
    # Average salary by department
    df.group_by("departm").agg(
        pl.col("salary").mean().alias("avg_salary"),
    ).sort("avg_salary", descending=True)
    return


@app.cell
def _(df, pl):
    # Multiple aggregations in a single call
    df.group_by("departm").agg(
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("salary").max().alias("max_salary"),
        pl.col("salary").min().alias("min_salary"),
        pl.col("publications").mean().alias("avg_publications"),
        pl.len().alias("count"),
    ).sort("avg_salary", descending=True)
    return


@app.cell
def _(df, pl):
    # Group by multiple columns
    df.group_by("departm", "gender").agg(
        pl.col("salary").mean().alias("avg_salary"),
        pl.len().alias("count"),
    ).sort("departm", "gender")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Window Functions

    Window functions compute values across groups **without collapsing rows**. In Polars, you use the `over()` expression to define the grouping. This is extremely powerful because it lets you add group-level statistics as new columns while keeping every individual row intact.

    This replaces the common pattern of grouping, computing a statistic, and then merging the result back to the original DataFrame.
    """)
    return


@app.cell
def _(df, pl):
    # Add each department's average salary as a column
    df.with_columns(
        pl.col("salary").mean().over("departm").alias("dept_avg_salary"),
    ).head(10)
    return


@app.cell
def _(df, pl):
    # Each person's salary as a percentage of their department's mean
    df_pct = df.with_columns(
        (pl.col("salary") / pl.col("salary").mean().over("departm") * 100)
        .round(1)
        .alias("pct_of_dept_mean"),
    )
    df_pct.sort("pct_of_dept_mean", descending=True).head(10)
    return


@app.cell
def _(df, pl):
    # Rank salary within each department
    df.with_columns(
        pl.col("salary").rank(descending=True).over("departm").alias("dept_salary_rank"),
    ).sort("departm", "dept_salary_rank").head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reshaping Data

    Polars provides `unpivot()` to go from wide to long format and `pivot()` to go from long to wide format. These are essential when preparing data for visualization or statistical modeling.
    """)
    return


@app.cell
def _(pl):
    # Create a wide-format example
    df_wide = pl.DataFrame({
        "department": ["bio", "chem", "neuro"],
        "q1_budget": [100, 200, 150],
        "q2_budget": [110, 190, 160],
        "q3_budget": [105, 210, 155],
    })
    df_wide
    return (df_wide,)


@app.cell
def _(df_wide):
    # Unpivot (wide to long): melt the quarterly columns into rows
    df_long = df_wide.unpivot(
        index="department",
        on=["q1_budget", "q2_budget", "q3_budget"],
        variable_name="quarter",
        value_name="budget",
    )
    df_long
    return (df_long,)


@app.cell
def _(df_long):
    # Pivot (long to wide): spread the quarter values back into columns
    df_long.pivot(
        on="quarter",
        index="department",
        values="budget",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lazy Evaluation

    One of Polars' most powerful features is **lazy evaluation**. Instead of executing each operation immediately, a LazyFrame records the operations as a query plan. Polars then optimizes this plan before execution, which can dramatically improve performance.

    Key optimizations that Polars applies automatically:

    - **Predicate pushdown**: Filters are pushed as early as possible, reducing the amount of data processed.
    - **Projection pushdown**: Only the columns you actually need are loaded from disk.
    - **Common subexpression elimination**: Repeated computations are calculated once.
    - **Parallel execution**: Independent operations run on multiple CPU cores.

    To use lazy mode, start with `pl.scan_csv()` instead of `pl.read_csv()`, or convert an existing DataFrame with `.lazy()`. When you are ready to execute, call `.collect()`.
    """)
    return


@app.cell
def _(pl, url):
    # Create a LazyFrame by scanning the CSV
    lf = pl.scan_csv(url)
    print(type(lf))
    lf
    return (lf,)


@app.cell
def _(lf, pl):
    # Build a query plan without executing it
    query = (
        lf
        .filter(pl.col("salary") > 60000)
        .group_by("departm")
        .agg(
            pl.col("salary").mean().alias("avg_salary"),
            pl.len().alias("count"),
        )
        .sort("avg_salary", descending=True)
    )

    # View the optimized query plan
    print(query.explain())
    return (query,)


@app.cell
def _(query):
    # Execute the query plan and get a DataFrame
    query.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also convert an existing DataFrame to a LazyFrame with `.lazy()` and back with `.collect()`. This is useful when you want to chain many operations and let Polars optimize them as a batch.
    """)
    return


@app.cell
def _(df, pl):
    # Convert eager DataFrame to lazy, apply operations, then collect
    result = (
        df.lazy()
        .with_columns(
            (pl.col("salary") / 1000).alias("salary_k"),
        )
        .filter(pl.col("salary_k") > 70)
        .select("departm", "salary_k", "publications")
        .collect()
    )
    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    Try these exercises to practice what you have learned. Each one uses the salary dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1**: Filter the salary data to only include rows where the `departm` is `"neuro"` and salary is above 80,000. How many rows match? What is the average salary of this subset?
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2**: Group the data by `departm` and `gender`. For each group, calculate the mean salary and the number of people. Sort the result by mean salary in descending order.
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3**: Using `over()`, create a new column called `pct_of_dept_mean` that shows each person's salary as a percentage of their department's mean salary. Sort by this column in descending order. Who has the highest relative salary compared to their department?
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    In this tutorial, we covered the core concepts of Polars:

    - **Series and DataFrames**: Polars' fundamental data structures with strict typing and no row index.
    - **Loading and inspecting data**: Reading CSVs, checking shapes, schemas, and null counts.
    - **Missing values**: Using `null_count()`, `drop_nulls()`, and `fill_null()`.
    - **The expression API**: Building computations with `pl.col()`, chaining operations, and using `when/then/otherwise`.
    - **Creating columns**: Adding new columns immutably with `with_columns()` and `.alias()`.
    - **Selecting and filtering**: Choosing columns with `select()` and rows with `filter()`.
    - **Operations**: Arithmetic, string methods, type casting, and `map_elements()`.
    - **Joins**: Combining DataFrames with `join()` and `concat()`.
    - **Grouping and aggregation**: Using `group_by().agg()` for summary statistics.
    - **Window functions**: Computing group-level values without collapsing rows using `over()`.
    - **Reshaping**: Converting between wide and long formats with `unpivot()` and `pivot()`.
    - **Lazy evaluation**: Building optimized query plans with `scan_csv()`, `.lazy()`, and `.collect()`.

    Polars' combination of speed, expressiveness, and lazy evaluation makes it an excellent tool for data analysis in Python. As your datasets grow larger and your queries more complex, these features will become increasingly valuable.
    """)
    return


if __name__ == "__main__":
    app.run()
