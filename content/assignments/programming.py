# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "f95b7d3b,a6050e8d,8af6cf8a,3c898569,0c2a9525,5b0811c2,4057a21d,8e573c96,bb431c42,f91c9868,47ea8b1a,3938720a,9a885e11,9ed8c312,623d28f9,88171b67,88d4f0d6,071c7218,2e048490,1abb7cde,66ff5488,bd3d1a89,6a6bcd39,80bcc5af"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "programming"
# grader-assignment-id = "33d31bd1-9b3d-44b0-9198-a74c1236dfbd"
# grader-assignment-version = "112a5f3c-8340-4323-a765-e9704ab30397"
# grader-version = "1"
# ///
"""DartBrains assignment: Introduction to Programming (instructor notebook).

Adapted from the Exercises at the end of the programming chapter:
https://dartbrains.org/Introduction_to_Programming/

Deliberately depends on nothing but marimo and the grader client -- the chapter
teaches plain Python, so the assignment should not require a scientific stack to
run. It starts fast in MoLab for the same reason.

Publish with (from dartbrains-grader/backend):
    uv run grader publish ../../dartbrains-assignments/assignments/programming.py \
        --server <server> --offering neuroimaging/2026-fall \
        --slug programming --title "Introduction to Programming"
"""

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
        # Assignment: Introduction to Programming

        These are the **Exercises** at the end of the
        [programming chapter](https://dartbrains.org/Introduction_to_Programming/) on
        dartbrains.org. Everything you need is in that chapter — work through it first, and
        keep it open in another tab.

        Each question has a cell for your answer, a **Check** cell that runs instantly and
        tells you what is still wrong, and a **Submit** button that records an attempt on the
        server. Sign in once at the top. Check as often as you like; it is not your grade.

        The last question is written, and an instructor reads it.
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
    _marks = {"prog-q01": 5, "prog-q02": 5, "prog-q03": 5, "prog-q04": 5, "prog-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. The even numbers

        Given the list below, build a new list called `evens` containing only its **even**
        elements, in the same order.

        ```python
        numbers = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
        ```

        *A number is even when the remainder of dividing it by two is zero.*
        """
    )
    return


@app.cell
def _():
    numbers = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

    evens = ...
    # YOUR CODE HERE
    pass
    return evens, numbers


@app.cell
def _(evens, g, mo, numbers):
    mo.stop(
        evens is ...,
        mo.md("**Complete the code cell above first.** This check runs once `evens` is defined."),
    )

    g.check(
        "prog-q01: The even numbers",
        [
            (isinstance(evens, list), "evens should be a list", 1),
            (
                all(n % 2 == 0 for n in evens) if isinstance(evens, list) else False,
                "every element of evens should be even",
                1,
            ),
            (evens == [4, 16, 36, 64, 100], "evens should be [4, 16, 36, 64, 100]", 2),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("prog-q01")
    return


@app.cell
def _(g):
    g.feedback("prog-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. The range of a list

        Write a function `value_range(values)` that takes a list of at least one number and
        returns the difference between the largest and the smallest.

        ```python
        value_range([17, 4, 9, 42, 25, 3])   # -> 39
        value_range([7])                      # -> 0
        ```
        """
    )
    return


@app.cell
def _():
    def value_range(values):
        # YOUR CODE HERE
        pass

    return (value_range,)


@app.cell
def _(g, mo, value_range):
    _probe = value_range([1, 2])
    mo.stop(
        _probe is None,
        mo.md("**Complete the function above first.** It should `return` a number."),
    )

    g.check(
        "prog-q02: The range of a list",
        [
            (value_range([17, 4, 9, 42, 25, 3]) == 39, "value_range([17, 4, 9, 42, 25, 3]) should be 39", 2),
            (value_range([7]) == 0, "a one-element list has a range of 0", 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("prog-q02")
    return


@app.cell
def _(g):
    g.feedback("prog-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. In both lists

        Find the numbers that appear in **both** lists below and put them in `in_both`,
        sorted smallest to largest.

        There is a list-comprehension answer and a one-line answer using sets. Either is
        fine — but if you use sets, remember a set has no order.
        """
    )
    return


@app.cell
def _():
    list_a = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361]
    list_b = [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]

    in_both = ...
    # YOUR CODE HERE
    pass
    return in_both, list_a, list_b


@app.cell
def _(g, in_both, mo):
    mo.stop(
        in_both is ...,
        mo.md("**Complete the code cell above first.** This check runs once `in_both` is defined."),
    )

    _expected = [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]
    g.check(
        "prog-q03: In both lists",
        [
            (isinstance(in_both, list), "in_both should be a list, not a set", 1),
            (
                sorted(in_both) == _expected if hasattr(in_both, "__iter__") else False,
                "in_both should contain exactly the ten numbers present in both lists",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("prog-q03")
    return


@app.cell
def _(g):
    g.feedback("prog-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. The speeding fine

        Write a function `speeding_fine(speed)` returning the fine in dollars:

        | speed (mph) | fine |
        |---|---|
        | 60 or below | `0` |
        | 61 to 80 inclusive | `100` |
        | 81 or above | `500` |

        *Watch the boundaries. 60, 61, 80 and 81 are the interesting cases, and the ones the
        checks look at.*
        """
    )
    return


@app.cell
def _():
    def speeding_fine(speed):
        # YOUR CODE HERE
        pass

    return (speeding_fine,)


@app.cell
def _(g, mo, speeding_fine):
    mo.stop(
        speeding_fine(70) is None,
        mo.md("**Complete the function above first.** It should `return` a number."),
    )

    g.check(
        "prog-q04: The speeding fine",
        [
            (speeding_fine(55) == 0, "55 mph should be $0", 1),
            (speeding_fine(70) == 100, "70 mph should be $100", 1),
            (speeding_fine(95) == 500, "95 mph should be $500", 1),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("prog-q04")
    return


@app.cell
def _(g):
    g.feedback("prog-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Reading an error

        Running this raises an error:

        ```python
        subject = {"id": "sub-01", "age": 24}
        print(subject["session"])
        ```

        In two or three sentences: name the error Python raises, explain what it is telling
        you, and say how you would change the code if a missing `"session"` is a normal thing
        to happen rather than a mistake.
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
    g.submit_button("prog-q05", outputs={"answer": answer.value})
    return


@app.cell
def _(g):
    g.feedback("prog-q05")
    return


if __name__ == "__main__":
    app.run()
