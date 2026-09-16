# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "polars", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "2377b2de,e8edc738,8af6cf8a,ca2cfd89,de18458e,d33a15ed,0f85b573,8c1266f1,eba26c71,9d483a7b,1cba575a,f543fdf2,5d41ff15,322e8c4e,5b1aecb3,b23a7007,a5706301,04d22c6a,0b23583c,f46e94d0,40952976,0acb3840,aefa4d65,4a6229f0,35610754,12d1def0"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "polars"
# grader-assignment-id = "dca34887-6a73-4623-83f4-52e64b472a2c"
# grader-assignment-version = "210b4537-2287-4f8d-9c8f-b6389cf3bb4e"
# grader-version = "1"
# ///
"""DartBrains assignment: Introduction to Polars."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return g, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Introduction to Polars

        These are the **Exercises** at the end of the
        [Polars chapter](https://dartbrains.org/Introduction_to_Polars/).

        This uses `salary.csv` — the dataset from the Polars chapter — and **not** the
        `salary_exercise.csv` from the pandas and plotting assignments. Different columns,
        different numbers; don't reuse answers from those.

        Every question here is auto-graded. Polars is strict about types, so a check that
        looks like it should pass but doesn't is usually telling you that something is a
        `LazyFrame` where a `DataFrame` was expected, or an `i64` where an `f64` was.

        Sign in once at the top.
        """
    )
    return


@app.cell
def _(g):
    g.signin_button()
    return


@app.cell
def _():
    # === MOGRADER: MARKS ===
    _marks = {"po-q01": 5, "po-q02": 5, "po-q03": 5, "po-q04": 5, "po-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        Run this cell as it is — every question uses `df`.

        77 rows, six columns: `salary`, `gender`, `departm`, `years`, `age`, `publications`.
        Note that `gender` is coded numerically here, not as text.
        """
    )
    return


@app.cell
def _(pl):
    URL = "https://raw.githubusercontent.com/ljchang/dartbrains/master/data/salary/salary.csv"
    df = pl.read_csv(URL)
    df.head()
    return URL, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Filtering on two conditions

        Make a DataFrame called `neuro_high` containing only the neuroscientists earning more
        than $80,000 — rows where `departm` is `"neuro"` **and** `salary` is above 80000.

        *Polars has no `and` keyword for expressions. Combine two `pl.col(...)` conditions with
        `&`, and parenthesise each one — `&` binds tighter than `>` does.*
        """
    )
    return


@app.cell
def _(df, pl):
    neuro_high = ...
    # YOUR CODE HERE
    pass
    neuro_high
    return (neuro_high,)


@app.cell
def _(g, mo, neuro_high):
    mo.stop(
        neuro_high is ...,
        mo.md("**Complete the code cell above first.** This check runs once `neuro_high` is defined."),
    )

    g.check(
        "po-q01: Filtering on two conditions",
        [
            (
                getattr(neuro_high, "height", None) == 6,
                "six people match both conditions",
                2,
            ),
            (
                abs(float(neuro_high["salary"].mean()) - 94599.6667) < 1.0,
                "their mean salary should be about $94,599.67",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("po-q01")
    return


@app.cell
def _(g):
    g.feedback("po-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Grouping by two columns

        Build `dept_gender`: one row per `departm`/`gender` combination, with the mean salary
        in a column named **`mean_salary`** and the number of people in a column named
        **`n`**, sorted by `mean_salary` from highest to lowest.

        *`group_by` accepts more than one column. Inside `.agg()`, `pl.len()` counts the rows in
        each group — it is the Polars equivalent of pandas' `size()`. Name each aggregation with
        `.alias()`, or the columns come back called `salary` and `len`.*
        """
    )
    return


@app.cell
def _(df, pl):
    dept_gender = ...
    # YOUR CODE HERE
    pass
    dept_gender
    return (dept_gender,)


@app.cell
def _(dept_gender, g, mo):
    mo.stop(
        dept_gender is ...,
        mo.md("**Complete the code cell above first.** This check runs once `dept_gender` is defined."),
    )

    _cols = set(getattr(dept_gender, "columns", []))
    g.check(
        "po-q02: Grouping by two columns",
        [
            (
                {"departm", "gender", "mean_salary", "n"} <= _cols,
                "the result needs columns departm, gender, mean_salary and n",
                1,
            ),
            (
                getattr(dept_gender, "height", None) == 13,
                "there are 13 department-by-gender combinations",
                2,
            ),
            (
                abs(float(dept_gender["mean_salary"][0]) - 79571.4615) < 1.0,
                "the top group's mean salary should be about $79,571.46",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("po-q02")
    return


@app.cell
def _(g):
    g.feedback("po-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. A window function

        Add a column called `pct_of_dept_mean` giving each person's salary as a percentage of
        their **own department's** mean salary, rounded to one decimal place. Call the result
        `salary_pct` and sort it by that column, highest first.

        This is the point of `over()`: you want a per-department number attached to every row
        **without collapsing the rows**, which is exactly what `group_by` would do to you.

        *`pl.col("salary").mean().over("departm")` computes the department mean and broadcasts
        it back across that department's rows. `.round(1)` and `.alias(...)` finish the job.*
        """
    )
    return


@app.cell
def _(df, pl):
    salary_pct = ...
    # YOUR CODE HERE
    pass
    salary_pct
    return (salary_pct,)


@app.cell
def _(g, mo, salary_pct):
    mo.stop(
        salary_pct is ...,
        mo.md("**Complete the code cell above first.** This check runs once `salary_pct` is defined."),
    )

    g.check(
        "po-q03: A window function",
        [
            (
                getattr(salary_pct, "height", None) == 77
                and "pct_of_dept_mean" in getattr(salary_pct, "columns", []),
                "all 77 rows should survive, with a new pct_of_dept_mean column",
                2,
            ),
            (
                abs(float(salary_pct["pct_of_dept_mean"][0]) - 158.3) < 0.15,
                "the highest relative salary should be about 158.3% of its department mean",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("po-q03")
    return


@app.cell
def _(g):
    g.feedback("po-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. Lazy evaluation

        Build the same query twice over, lazily.

        Starting from `pl.scan_csv(URL)` — **not** `pl.read_csv` — keep only people earning more
        than 60000, then compute per department the mean salary as `avg_salary` and the head
        count as `count`, sorted by `avg_salary` descending.

        Assign the **unexecuted** query plan to `lazy_query`, then assign the executed result to
        `lazy_result`.

        ```python
        lazy_query  = pl.scan_csv(URL).filter(...)...   # still a LazyFrame
        lazy_result = lazy_query.collect()              # now a DataFrame
        ```

        *`scan_csv` returns a `LazyFrame` and nothing runs until you call `.collect()`. If you are
        curious what Polars did with your query before running it, print
        `lazy_query.explain()` — you should see the filter pushed down into the CSV scan.*
        """
    )
    return


@app.cell
def _(URL, pl):
    lazy_query = ...
    lazy_result = ...
    # YOUR CODE HERE
    pass
    lazy_result
    return lazy_query, lazy_result


@app.cell
def _(g, lazy_query, lazy_result, mo, pl):
    mo.stop(
        lazy_query is ... or lazy_result is ...,
        mo.md("**Complete the code cell above first.** This check runs once `lazy_query` and `lazy_result` are defined."),
    )

    g.check(
        "po-q04: Lazy evaluation",
        [
            (
                isinstance(lazy_query, pl.LazyFrame),
                "lazy_query should still be a LazyFrame -- do not call .collect() on it",
                2,
            ),
            (
                isinstance(lazy_result, pl.DataFrame) and lazy_result.height == 7,
                "lazy_result should be a collected DataFrame with one row per department (7)",
                1,
            ),
            (
                abs(float(lazy_result["avg_salary"][0]) - 81291.4167) < 1.0,
                "the best-paid department above the cutoff averages about $81,291.42",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("po-q04")
    return


@app.cell
def _(g):
    g.feedback("po-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Missing values, and what dropping them costs

        This dataset has very few nulls, which makes it a good place to see why the *cheap* fix
        is rarely the right one. Three parts:

        1. `null_counts` — a one-row DataFrame of the number of nulls in each column.
        2. `complete_only` — `df` with every row containing **any** null removed.
        3. `years_filled` — `df` with a new column **`years_imputed`**, equal to `years` but with
           nulls replaced by the **median `years` of that person's own department**.

        Part 3 is the interesting one: it reuses `over()` from Q3. A null `years` is not a reason
        to throw away a perfectly good `salary`, and imputing from the department is a better
        guess than imputing from the whole dataset.

        *`.null_count()` for part 1, `.drop_nulls()` for part 2. For part 3, `fill_null()` takes an
        expression, so you can hand it `pl.col("years").median().over("departm")`.*
        """
    )
    return


@app.cell
def _(df, pl):
    null_counts = ...
    complete_only = ...
    years_filled = ...
    # YOUR CODE HERE
    pass
    null_counts
    return complete_only, null_counts, years_filled


@app.cell
def _(complete_only, g, mo, null_counts, years_filled):
    mo.stop(
        null_counts is ... or complete_only is ... or years_filled is ...,
        mo.md("**Complete the code cell above first.** This check runs once all three names are defined."),
    )

    g.check(
        "po-q05: Missing values, and what dropping them costs",
        [
            (
                int(null_counts["years"][0]) == 1 and int(null_counts["age"][0]) == 1,
                "there is exactly one null in years and one in age",
                1,
            ),
            (
                getattr(complete_only, "height", None) == 75,
                "dropping every row with any null costs you 2 of the 77 rows",
                1,
            ),
            (
                "years_imputed" in getattr(years_filled, "columns", [])
                and int(years_filled["years_imputed"].null_count()) == 0,
                "years_imputed should have no nulls left",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("po-q05")
    return


@app.cell
def _(g):
    g.feedback("po-q05")
    return


if __name__ == "__main__":
    app.run()
