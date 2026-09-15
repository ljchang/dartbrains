# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "3c706731,3b71c6dc,8af6cf8a,5e99cf5a,66e7d653,205bdc96,9e72bcd6,5326dd3b,469ab2b6,93ea6ceb,8dbbe498,6f01a8d3,f04c8b32,fa9bf782,78004e3e,851d2086,616918e3,e27baa7a,a13f65b3,8a99d3b3,1217d5cc,bd3d1a89,78d7c020,c0244584"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "pandas"
# grader-assignment-id = "b6b9c33d-5d09-4280-b937-4ec9d65b67d7"
# grader-assignment-version = "98713443-5c16-46c0-9e18-8a12581da0f3"
# grader-version = "1"
# ///
"""DartBrains assignment: Introduction to Pandas (instructor notebook).

Adapted from the Exercises at the end of the pandas chapter:
https://dartbrains.org/Introduction_to_Pandas/

Uses salary_exercise.csv (Weisberg 1985): 52 tenure-track professors, three rows
with a missing value. Note the column *values* are strings ("male", "doctorate"),
not the 0/1 codes the old exercise text described -- the prose had drifted from
the data.

Publish with (from dartbrains-grader/backend):
    uv run grader publish ../../dartbrains-assignments/assignments/pandas-intro.py \
        --server <server> --offering neuroimaging/2026-fall \
        --slug pandas --title "Introduction to Pandas"
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return g, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Introduction to Pandas

        These are the **Exercises** at the end of the
        [pandas chapter](https://dartbrains.org/Introduction_to_Pandas/) on dartbrains.org.

        The data are from Weisberg (1985): salaries for 52 tenure-track professors at a small
        college. The raw columns are terse, and three rows have a missing value.

        | column | meaning |
        |---|---|
        | `sx` | sex |
        | `rk` | rank — assistant, associate, or full |
        | `yr` | years in current rank |
        | `dg` | highest degree — doctorate or masters |
        | `yd` | years since that degree |
        | `sl` | academic-year salary, in dollars |

        Each question has a cell for your code, a **Check** cell that runs instantly, and a
        **Submit** button. Sign in once at the top. The last question is written, and an
        instructor reads it.
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
    _marks = {"pd-q01": 5, "pd-q02": 5, "pd-q03": 5, "pd-q04": 5, "pd-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Load and clean

        Read the file into a dataframe called `salary`, rename the columns to
        `sex`, `rank`, `yearsinrank`, `degree`, `yearssinceHD`, `salary`, in that order, and
        drop every row that has a missing value.

        *`pd.read_csv` takes a URL. `.rename(columns=...)` takes a dict. `.dropna()` removes
        rows with missing values and returns a new dataframe — it does not change the old one
        unless you ask it to.*
        """
    )
    return


@app.cell
def _(pd):
    URL = "https://raw.githubusercontent.com/ljchang/dartbrains/master/data/salary/salary_exercise.csv"

    salary = ...
    # YOUR CODE HERE
    pass
    return URL, salary


@app.cell
def _(g, mo, salary):
    mo.stop(
        salary is ...,
        mo.md("**Complete the code cell above first.** This check runs once `salary` is defined."),
    )

    _cols = ["sex", "rank", "yearsinrank", "degree", "yearssinceHD", "salary"]
    g.check(
        "pd-q01: Load and clean",
        [
            (hasattr(salary, "columns"), "salary should be a pandas DataFrame", 1),
            (list(getattr(salary, "columns", [])) == _cols, f"the columns should be {_cols}", 2),
            (len(salary) == 49, "49 of the 52 rows survive dropping the missing values", 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pd-q01")
    return


@app.cell
def _(g):
    g.feedback("pd-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Describe the salaries

        Put the overall **mean**, **standard deviation**, **minimum** and **maximum** of
        `salary` into a dictionary called `salary_stats`, with exactly those four keys:

        ```python
        salary_stats = {"mean": ..., "std": ..., "min": ..., "max": ...}
        ```

        *`.describe()` gives you all four at once if you would rather read them off.*
        """
    )
    return


@app.cell
def _(salary):
    salary_stats = ...
    # YOUR CODE HERE
    pass
    return (salary_stats,)


@app.cell
def _(g, mo, salary_stats):
    mo.stop(
        salary_stats is ...,
        mo.md("**Complete the code cell above first.**"),
    )

    _s = salary_stats if isinstance(salary_stats, dict) else {}
    g.check(
        "pd-q02: Describe the salaries",
        [
            (
                set(_s) == {"mean", "std", "min", "max"},
                "salary_stats needs exactly the keys mean, std, min, max",
                1,
            ),
            (abs(float(_s.get("mean", 0)) - 23895.08) < 0.5, "the mean is about 23895.08", 1),
            (abs(float(_s.get("min", 0)) - 15000) < 0.5, "the minimum is 15000", 1),
            (abs(float(_s.get("max", 0)) - 38045) < 0.5, "the maximum is 38045", 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pd-q02")
    return


@app.cell
def _(g):
    g.feedback("pd-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. The most senior professors of each degree

        For each degree type, take the **5 professors with the most years since their
        highest degree** (`yearssinceHD`) and compute their mean salary. Put the answer in
        `senior_means` as a dict keyed by degree:

        ```python
        {"doctorate": ..., "masters": ...}
        ```

        *Sorting first and then taking `.head(5)` of each group is one way. A Series with the
        degrees as its index converts with `.to_dict()`.*
        """
    )
    return


@app.cell
def _(salary):
    senior_means = ...
    # YOUR CODE HERE
    pass
    return (senior_means,)


@app.cell
def _(g, mo, senior_means):
    mo.stop(senior_means is ..., mo.md("**Complete the code cell above first.**"))

    _s = dict(senior_means) if hasattr(senior_means, "keys") else {}
    g.check(
        "pd-q03: The most senior professors of each degree",
        [
            (set(_s) == {"doctorate", "masters"}, "senior_means should be keyed by degree", 1),
            (
                abs(float(_s.get("doctorate", 0)) - 31023.6) < 1.0,
                "the five most senior doctorates average about 31023.60",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pd-q03")
    return


@app.cell
def _(g):
    g.feedback("pd-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. Salary by rank, and by sex

        Two groupings, so the next question has something to work with:

        - `by_rank` — mean salary for each `rank`, as a dict
        - `by_sex` — mean salary for each `sex`, as a dict
        """
    )
    return


@app.cell
def _(salary):
    by_rank = ...
    by_sex = ...
    # YOUR CODE HERE
    pass
    return by_rank, by_sex


@app.cell
def _(by_rank, by_sex, g, mo):
    mo.stop(
        by_rank is ... or by_sex is ...,
        mo.md("**Complete the code cell above first.**"),
    )

    _r = dict(by_rank) if hasattr(by_rank, "keys") else {}
    _x = dict(by_sex) if hasattr(by_sex, "keys") else {}
    g.check(
        "pd-q04: Salary by rank, and by sex",
        [
            (set(_r) == {"assistant", "associate", "full"}, "by_rank should have the three ranks", 1),
            (abs(float(_r.get("full", 0)) - 29658.95) < 1.0, "full professors average about 29658.95", 1),
            (set(_x) == {"female", "male"}, "by_sex should have both sexes", 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pd-q04")
    return


@app.cell
def _(g):
    g.feedback("pd-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Interpretation

        Across everyone, men in these data earn about **\$3,553** more per year than women.
        Grouped by rank as well, the difference is much smaller:

        | rank | female | male | difference |
        |---|---|---|---|
        | assistant | 17,580 | 17,902 | 322 |
        | associate | 21,570 | 23,278 | 1,708 |
        | full | 28,805 | 29,872 | 1,067 |

        In three or four sentences: explain how the overall difference can be so much larger
        than any of the within-rank differences, say what that implies about reporting the
        overall number on its own, and name one thing you would want to know about how people
        reach each rank before concluding anything about pay.
        """
    )
    return


@app.cell
def _(mo):
    answer = mo.ui.text_area(placeholder="Your answer...", full_width=True)
    answer
    return (answer,)


@app.cell
def _(answer, g):
    # UI element values are not part of the notebook file, so pass them as outputs;
    # they are stored with the attempt and shown to the grader next to the notebook.
    g.submit_button("pd-q05", outputs={"answer": answer.value})
    return


@app.cell
def _(g):
    g.feedback("pd-q05")
    return


if __name__ == "__main__":
    app.run()
