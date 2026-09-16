# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "f95b7d3b,16d2aaba,8af6cf8a,1b2680f5,946f1a5c,5bb74da5,a418a6b7,a390382a,fa78be7b,35b60527,3f2e1145,44587398,30ec26c6,574c5c99,c740282d,783dc3ff"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "practice"
# grader-assignment-id = "ac60469b-65b6-4b07-90d8-2cfbf7d37e00"
# grader-assignment-version = "c53dd127-588a-42d9-9f8a-8f38df227a59"
# grader-version = "1"
# ///
"""DartBrains assignment: Practice — how submitting works."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return g, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Practice: how submitting works

        This one does not count for anything. It exists so you can walk the whole loop —
        sign in, **Check**, **Submit**, read your **Feedback** — before any of it matters.

        Every answer is written out for you in the question. Copy it in. The point is to
        watch the machinery work, not to work anything out.

        Do this once now. If your Dartmouth sign-in is going to give you trouble, it is
        much better to find out here than an hour before a real assignment is due.

        The guide that walks through each step is
        [Submitting assignments](https://dartbrains.org/assignments/how-to-submit/).
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
    _marks = {"prac-q01": 1, "prac-q02": 1, "prac-q03": 1}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Write down a number

        Set `answer` to `42`.

        ```python
        answer = 42
        ```

        That is the whole question. Type it into the cell below and watch the **Check**
        cell underneath go green the moment you do — it re-runs by itself, with nothing
        to press.
        """
    )
    return


@app.cell
def _():
    answer = ...
    # YOUR CODE HERE
    pass
    return (answer,)


@app.cell
def _(answer, g, mo):
    mo.stop(
        answer is ...,
        mo.md("**Set `answer` above first.** This check runs on its own once you do."),
    )

    g.check(
        "prac-q01: Write down a number",
        [(answer == 42, "answer should be 42", 1)],
    )
    return


@app.cell
def _(g):
    g.submit_button("prac-q01")
    return


@app.cell
def _(g):
    g.feedback("prac-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. A function, and the tests you cannot see

        Write a function `double(x)` that returns twice `x`:

        ```python
        def double(x):
            return 2 * x
        ```

        This question is here to show you something about **Check**. The check below runs
        one test, on `double(4)`. When you press **Submit**, the grader runs that test
        *and* two more that are not in your notebook.

        That is why a green check is not a grade. It means you have not made an obvious
        mistake — nothing more. Here the hidden tests are as easy as the visible one; in a
        real assignment they are where the corner cases live.
        """
    )
    return


@app.cell
def _():
    def double(x):
        # YOUR CODE HERE
        pass

    return (double,)


@app.cell
def _(double, g, mo):
    _probe = double(1)
    mo.stop(
        _probe is None,
        mo.md("**Complete the function above first.** It should `return` a number."),
    )

    g.check(
        "prac-q02: A function, and the tests you cannot see",
        [
            (double(4) == 8, "double(4) should be 8", 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("prac-q02")
    return


@app.cell
def _(g):
    g.feedback("prac-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. A question a person reads

        Not every question can be marked by a machine. This one is read by an instructor,
        and the difference is worth seeing once before it matters.

        **In a sentence or two: what happened when you pressed Submit on Q1 or Q2?** Say
        what the Feedback cell showed you, and roughly how long it took.

        Anything you actually observed is a fine answer. When you submit this one, the
        Feedback cell will say it is waiting for a grade rather than giving you a score —
        that is correct, and it stays that way until someone reads it.
        """
    )
    return


@app.cell
def _(mo):
    answer_text = mo.ui.text_area(placeholder="Your answer...", full_width=True)
    answer_text
    return (answer_text,)


@app.cell
def _(answer_text, g):
    # UI element values are not part of the notebook file, so pass them as outputs;
    # they are stored with the attempt and shown to the grader next to the notebook.
    g.submit_button("prac-q03", outputs={"answer": answer_text.value})
    return


@app.cell
def _(g):
    g.feedback("prac-q03")
    return


if __name__ == "__main__":
    app.run()
