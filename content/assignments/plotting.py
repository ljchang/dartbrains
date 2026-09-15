# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "seaborn", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "2e2dede1,11f84f99,8af6cf8a,d976c04a,c40541f3,9ad0082b,7179c7d6,72ebc644,43f1e986,e3843616,43d14854,914b1c65,1e3e3791,c8af50cc,95978044,8a565344,1c441b09,2dfbc62b,fecd5238,4ebae1ac,32ed5d21,7b3bb2da,9c85a9cf,bd3d1a89,c0db07db,3b37f974"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "plotting"
# grader-assignment-id = "3c36bb36-7272-4a3f-8bd2-74cfe4c69bcb"
# grader-assignment-version = "e8459aa3-6b3f-4b6b-952b-b57be49bc108"
# grader-version = "1"
# ///
"""DartBrains assignment: Introduction to Plotting (instructor notebook).

Adapted from the Exercises at the end of the plotting chapter:
https://dartbrains.org/Introduction_to_Plotting/

Grading a figure is not the same as grading a number. The checks here look at
two things that are actually well defined: the data behind the plot, and the
structure of the figure object -- how many axes, what the labels say, whether an
artist of the right kind was drawn. Aesthetic faithfulness to the reference
image stays ungraded practice, because "looks the same" has no threshold.

Publish with (from dartbrains-grader/backend):
    uv run grader publish ../../dartbrains-assignments/assignments/plotting.py \
        --server <server> --offering neuroimaging/2026-fall \
        --slug plotting --title "Introduction to Plotting"
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return g, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Introduction to Plotting

        These are the **Exercises** at the end of the
        [plotting chapter](https://dartbrains.org/Introduction_to_Plotting/), on the same
        Weisberg salary data as the pandas assignment.

        A note on how these are graded. A figure has no single right answer, so the checks
        look at the two things that are well defined: **the numbers behind the plot**, and
        **the structure of the figure** — how many axes it has, what the labels say, whether
        the right kind of artist was drawn. Colour, font and spacing are yours.

        Sign in once at the top. The last question is written.
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
    _marks = {"pl-q01": 5, "pl-q02": 5, "pl-q03": 5, "pl-q04": 5, "pl-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        Run this cell as it is — every question uses `salary`.
        """
    )
    return


@app.cell
def _(pd):
    URL = "https://raw.githubusercontent.com/ljchang/dartbrains/master/data/salary/salary_exercise.csv"
    salary = pd.read_csv(URL).rename(
        columns={
            "sx": "sex",
            "rk": "rank",
            "yr": "yearsinrank",
            "dg": "degree",
            "yd": "yearssinceHD",
            "sl": "salary",
        }
    ).dropna()
    salary.head()
    return URL, salary


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. The numbers behind the heatmap

        Before drawing anything, compute what you are going to draw. Make two correlation
        matrices of the numeric columns — one for the male professors, one for the female —
        as `corr_male` and `corr_female`.

        *`.corr(numeric_only=True)` on a filtered frame. There are three numeric columns.*
        """
    )
    return


@app.cell
def _(salary):
    corr_male = ...
    corr_female = ...
    # YOUR CODE HERE
    pass
    return corr_female, corr_male


@app.cell
def _(corr_female, corr_male, g, mo):
    mo.stop(
        corr_male is ... or corr_female is ...,
        mo.md("**Complete the code cell above first.**"),
    )

    g.check(
        "pl-q01: The numbers behind the heatmap",
        [
            (
                getattr(corr_male, "shape", None) == (3, 3)
                and getattr(corr_female, "shape", None) == (3, 3),
                "each correlation matrix should be 3x3 -- the three numeric columns",
                1,
            ),
            (
                abs(float(corr_male.loc["yearsinrank", "salary"]) - 0.7342) < 0.005,
                "for men, years in rank correlates with salary at about 0.73",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pl-q01")
    return


@app.cell
def _(g):
    g.feedback("pl-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Two heatmaps side by side

        Draw them. Make a figure called `fig_heat` with **two axes side by side**, a heatmap
        on each, titled `"Male"` on the left and `"Female"` on the right.

        Use a diverging colormap and fix the scale to run from −1 to 1 — otherwise the two
        panels use different scales and cannot be compared, which is the whole point of
        putting them next to each other.
        """
    )
    return


@app.cell
def _(corr_female, corr_male, plt, sns):
    fig_heat = ...
    # YOUR CODE HERE
    pass
    fig_heat
    return (fig_heat,)


@app.cell
def _(fig_heat, g, mo):
    mo.stop(fig_heat is ..., mo.md("**Complete the code cell above first.**"))

    _axes = [a for a in getattr(fig_heat, "axes", []) if a.get_title()]
    _titles = [a.get_title() for a in _axes]
    g.check(
        "pl-q02: Two heatmaps side by side",
        [
            (hasattr(fig_heat, "axes"), "fig_heat should be a matplotlib Figure", 1),
            (
                sorted(t.lower() for t in _titles) == ["female", "male"],
                'the two panels should be titled "Male" and "Female"',
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pl-q02")
    return


@app.cell
def _(g):
    g.feedback("pl-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. Mean salary by sex, with error bars

        Make a figure `fig_bar` with one axes: a bar for the mean salary of each sex, with an
        error bar showing the **standard error of the mean** (the standard deviation divided
        by the square root of the group size).

        Label the y-axis `"Salary"`.
        """
    )
    return


@app.cell
def _(plt, salary):
    fig_bar = ...
    # YOUR CODE HERE
    pass
    fig_bar
    return (fig_bar,)


@app.cell
def _(fig_bar, g, mo):
    mo.stop(fig_bar is ..., mo.md("**Complete the code cell above first.**"))

    _ax = fig_bar.axes[0] if getattr(fig_bar, "axes", None) else None
    _heights = sorted(p.get_height() for p in getattr(_ax, "patches", []))
    g.check(
        "pl-q03: Mean salary by sex, with error bars",
        [
            (_ax is not None and len(_ax.patches) == 2, "there should be two bars", 1),
            (
                len(_heights) == 2 and abs(_heights[0] - 21357.14) < 1.0,
                "the shorter bar is the female mean, about 21357",
                1,
            ),
            ((_ax.get_ylabel() or "").lower() == "salary", 'label the y-axis "Salary"', 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pl-q03")
    return


@app.cell
def _(g):
    g.feedback("pl-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. Salary against years in rank

        Make a figure `fig_scatter` with one axes: a scatterplot of `salary` on the y-axis
        against `yearsinrank` on the x-axis, with both axes labelled.

        Every professor should appear — all 49 of them.
        """
    )
    return


@app.cell
def _(plt, salary):
    fig_scatter = ...
    # YOUR CODE HERE
    pass
    fig_scatter
    return (fig_scatter,)


@app.cell
def _(fig_scatter, g, mo):
    mo.stop(fig_scatter is ..., mo.md("**Complete the code cell above first.**"))

    _ax = fig_scatter.axes[0] if getattr(fig_scatter, "axes", None) else None
    _n = sum(len(c.get_offsets()) for c in getattr(_ax, "collections", []))
    g.check(
        "pl-q04: Salary against years in rank",
        [
            (_ax is not None and len(_ax.collections) > 0, "there should be a scatterplot", 1),
            (_n == 49, f"all 49 professors should be plotted (found {_n})", 2),
            (
                bool(_ax.get_xlabel()) and bool(_ax.get_ylabel()),
                "label both axes",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("pl-q04")
    return


@app.cell
def _(g):
    g.feedback("pl-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. What the bar chart hides

        Your Q3 bar chart shows two means and two error bars — four numbers standing in for
        49 people. Your Q4 scatterplot shows all of them.

        In three or four sentences: say what the bar chart cannot show that the scatterplot
        can, name a distribution of salaries that would produce exactly the same bar chart
        but tell a very different story, and say which of the two you would put in a paper
        reporting a pay gap, and why.
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
    g.submit_button("pl-q05", outputs={"answer": answer.value})
    return


@app.cell
def _(g):
    g.feedback("pl-q05")
    return


if __name__ == "__main__":
    app.run()
