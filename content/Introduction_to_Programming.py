import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Programming
    _Written by Luke Chang_

    This notebook runs **in your browser**. There is nothing to install: every cell below is
    live Python, and you are encouraged to change the code and see what happens. Breaking a
    cell costs nothing — reload the page and it comes back.

    If you want to run Python on your own machine, see
    **[Setting up Python](Setting_Up_Python.html)** first.

    A note on how to read this page. Many cells have a slider or a box above them. Those are
    not decoration — move one and every cell that depends on it re-runs immediately. That is
    what a *reactive* notebook means, and it is the fastest way to build intuition about what
    a piece of code actually does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variables and types

    A variable is a name bound to a value. Python works out the type for you, and `type()`
    tells you what it decided.

    Type a Python value into the box — try `42`, `3.14`, `'hello'`, `True`, `None`,
    `[1, 2, 3]`, `{'a': 1}` — and watch what Python makes of it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    literal = mo.ui.text(value="42", label="A Python value:", full_width=False)
    literal
    return (literal,)


@app.cell(hide_code=True)
def _(literal, mo):
    try:
        _value = eval(literal.value, {"__builtins__": {}}, {})
        _t = type(_value).__name__
        _out = mo.md(f"""
        ```python
        x = {literal.value}
        type(x)   # -> {_t}
        ```
        Python read that as a **`{_t}`**, with the value `{_value!r}`.
        """)
    except Exception as _e:
        _out = mo.md(f"""
        ```python
        x = {literal.value}
        ```
        That is not valid Python: **{type(_e).__name__}** — {_e}
        """)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The types you will meet constantly:

    | Type | Example | What it is |
    |---|---|---|
    | `int` | `42` | a whole number |
    | `float` | `3.14` | a number with a decimal point |
    | `str` | `'hello'` | text |
    | `bool` | `True` | true or false |
    | `NoneType` | `None` | "no value" — not zero, not empty |

    You can convert between them when the conversion makes sense. `str(1)` gives `'1'`, and
    `int('1')` gives `1` — but `int('hello')` raises a `ValueError`, because there is no
    sensible answer.
    """)
    return


@app.cell
def _():
    a = 1
    b = 1.0
    c = "hello"
    d = True
    e = None

    print(f"{a!r:>8}  is {type(a).__name__}")
    print(f"{b!r:>8}  is {type(b).__name__}")
    print(f"{c!r:>8}  is {type(c).__name__}")
    print(f"{d!r:>8}  is {type(d).__name__}")
    print(f"{e!r:>8}  is {type(e).__name__}")
    print()
    print(f"str(1) -> {str(1)!r}      int('1') -> {int('1')!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arithmetic

    The usual operators, plus two that surprise people. Change the numbers and the operator:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lhs = mo.ui.number(value=7, start=-100, stop=100, label="")
    op = mo.ui.dropdown(
        options=["+", "-", "*", "/", "//", "%", "**"], value="/", label=""
    )
    rhs = mo.ui.number(value=2, start=-100, stop=100, label="")
    mo.hstack([lhs, op, rhs], justify="start", gap=1)
    return lhs, op, rhs


@app.cell(hide_code=True)
def _(lhs, mo, op, rhs):
    _expr = f"{lhs.value} {op.value} {rhs.value}"
    _notes = {
        "/": "`/` is **true division** and always gives a `float`, even when it divides evenly.",
        "//": "`//` is **floor division** — it divides and rounds *down* to a whole number.",
        "%": "`%` is the **modulo**: the remainder. `n % 2 == 0` is the usual test for even.",
        "**": "`**` is exponentiation, not `^`. In Python `^` means something else entirely.",
    }
    try:
        _result = eval(_expr, {"__builtins__": {}}, {})
        _body = f"""
        ```python
        {_expr}
        ```
        → `{_result!r}`  (a `{type(_result).__name__}`)
        """
    except Exception as _err:
        _body = f"""
        ```python
        {_expr}
        ```
        raises **{type(_err).__name__}**: {_err}
        """
    mo.md(_body + ("\n" + _notes[op.value] if op.value in _notes else ""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Strings

    `+` joins strings and `*` repeats them. The same symbols do different things depending on
    the type — adding numbers and adding strings are not the same operation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    word = mo.ui.text(value="ha", label="text:")
    times = mo.ui.slider(1, 12, value=3, label="repeat:", show_value=True)
    mo.hstack([word, times], justify="start", gap=2)
    return times, word


@app.cell(hide_code=True)
def _(mo, times, word):
    mo.md(f"""
    ```python
    word = {word.value!r}
    word * {times.value}        # -> {word.value * times.value!r}
    word + "!"       # -> {word.value + "!"!r}
    len(word)        # -> {len(word.value)}
    word.upper()     # -> {word.value.upper()!r}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Putting values into text

    Nearly every `print` on this page uses an **f-string**: a string prefixed with `f`, in
    which anything inside `{braces}` is evaluated and dropped into the text.

    ```python
    name, n = "Luke", 3
    f"{name} ran {n} subjects"    # -> 'Luke ran 3 subjects'
    f"{n} squared is {n ** 2}"    # -> '3 squared is 9'
    ```

    A colon introduces formatting, which is how you stop a float printing to seventeen
    decimal places:

    ```python
    f"{3.14159:.2f}"   # -> '3.14'     two decimal places
    f"{42:>6}"         # -> '    42'   right-aligned in six columns
    f"{0.87:.1%}"      # -> '87.0%'    as a percentage
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparisons and logic

    Comparisons produce a `bool`. `and`, `or` and `not` combine them.

    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `=` **assigns**, `==` **compares**. `x = 5` sets `x` to five; `x == 5` asks whether it
    already is. Python will not stop you writing one where you meant the other.
    """).callout(kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    left = mo.ui.number(value=5, start=-20, stop=20, label="")
    comp = mo.ui.dropdown(options=["==", "!=", "<", "<=", ">", ">="], value="<", label="")
    right = mo.ui.number(value=10, start=-20, stop=20, label="")
    mo.hstack([left, comp, right], justify="start", gap=1)
    return comp, left, right


@app.cell(hide_code=True)
def _(comp, left, mo, right):
    _expr = f"{left.value} {comp.value} {right.value}"
    _val = eval(_expr, {"__builtins__": {}}, {})
    mo.md(f"""
    ```python
    {_expr}
    ```
    → **`{_val}`**

    Combining them: `({_expr}) and ({left.value} != 0)` → `{_val and left.value != 0}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conditional logic

    `if` runs a block when a condition is true, `elif` offers another condition, and `else`
    catches everything remaining. Python decides where a block begins and ends by
    **indentation** — there are no braces, and the indentation is not cosmetic.

    Drag the speed and watch which branch runs. This is the same logic as the speeding-ticket
    exercise at the end of the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    speed = mo.ui.slider(40, 110, value=75, label="speed (mph):", show_value=True)
    speed
    return (speed,)


@app.cell(hide_code=True)
def _(mo, speed):
    # Rendered as HTML, not a fenced code block: a fence cannot carry a highlight,
    # and the whole point of this cell is showing *which line runs*.
    _s = speed.value
    _active = 0 if _s <= 60 else 1 if _s <= 80 else 2
    _fine = (0, 100, 500)[_active]

    _rows = "".join(
        "<div style='padding:1px 10px;border-left:3px solid "
        + (
            "#00693e;background:#00693e14'>"
            if _group == _active
            else "transparent;opacity:.38'>"
        )
        + _line
        + "</div>"
        for _line, _group in [
            ("if speed &lt;= 60:", 0),
            ("    fine = 0", 0),
            ("elif speed &lt;= 80:", 1),
            ("    fine = 100", 1),
            ("else:", 2),
            ("    fine = 500", 2),
        ]
    )
    mo.Html(
        f"""
        <div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;
                    line-height:1.6;white-space:pre;margin:.4rem 0">
          <div style="opacity:.5;padding:1px 10px">speed = {_s}</div>
          {_rows}
        </div>
        <p style="margin:.4rem 0 0">The highlighted branch is the one that runs, so
        <code>fine</code> is <b>${_fine}</b>.</p>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loops

    A `for` loop walks over the items of something. A `while` loop keeps going until its
    condition stops being true.

    `range(n)` produces the numbers `0` to `n-1` — note it stops *before* `n`, which is the
    same convention as slicing below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n = mo.ui.slider(1, 12, value=5, label="n:", show_value=True)
    n
    return (n,)


@app.cell(hide_code=True)
def _(mo, n):
    _squares = [i**2 for i in range(n.value)]
    _lines = "\n".join(f"  i = {i:<3} i**2 = {i**2}" for i in range(n.value))
    mo.md(f"""
    ```python
    for i in range({n.value}):
        print(i, i ** 2)
    ```
    ```text
    {_lines}
    ```
    The same thing as a **list comprehension**, which is the idiomatic way to build a list
    from a loop:
    ```python
    [i ** 2 for i in range({n.value})]   # -> {_squares}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions

    A function packages a piece of work under a name so you can use it more than once.
    `def` names it, the parameters are its inputs, and `return` hands a value back.

    Edit this one — change the rates, add a branch — and the cells below re-run.
    """)
    return


@app.cell
def _():
    def speeding_fine(speed):
        """Return the fine in dollars for a given speed in mph."""
        if speed <= 60:
            return 0
        elif speed <= 80:
            return 100
        return 500

    return (speeding_fine,)


@app.cell(hide_code=True)
def _(mo, speeding_fine):
    _rows = "\n".join(
        f"| {s} | ${speeding_fine(s)} |" for s in (55, 60, 61, 80, 81, 100)
    )
    mo.md(f"""
    | speed | fine |
    |---|---|
    {_rows}

    Because the notebook is reactive, editing `speeding_fine` above rewrites this table
    immediately — you never re-run anything by hand.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lists

    A list is an ordered, changeable sequence. Python counts from **0**, so the first item is
    `a[0]`.

    Slicing is `a[start:stop:step]`, and it includes `start` but **excludes** `stop`. That
    off-by-one is the single most common source of confusion for beginners, so rather than
    explain it again, move the sliders and watch which items survive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    start = mo.ui.slider(-10, 10, value=0, label="start:", show_value=True)
    stop = mo.ui.slider(-10, 10, value=10, label="stop:", show_value=True)
    step = mo.ui.slider(-3, 3, value=1, label="step:", show_value=True)
    mo.hstack([start, stop, step], justify="start", gap=2)
    return start, step, stop


@app.cell(hide_code=True)
def _(mo, start, step, stop):
    _a = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    _step = step.value or 1  # a step of 0 is an error in Python
    _sliced = _a[start.value : stop.value : _step]
    _kept = set(id(x) for x in _sliced)
    _shown = "  ".join(
        f"**{v}**" if v in _sliced else f"<span style='opacity:.3'>{v}</span>"
        for v in _a
    )
    mo.md(f"""
    ```python
    a = {_a}
    a[{start.value}:{stop.value}:{_step}]
    ```
    {_shown}

    → `{_sliced}`

    {"A negative step walks backwards." if _step < 0 else ""}
    {"An empty result: with this step, start never reaches stop." if not _sliced else ""}
    """)
    return


@app.cell
def _():
    squares = [1, 4, 9, 16, 25]

    squares.append(36)  # add to the end
    squares[0] = 100  # lists are mutable: you can change an item in place

    print(squares)
    print(f"length {len(squares)}, last {squares[-1]}, first three {squares[:3]}")
    print(f"evens only: {[v for v in squares if v % 2 == 0]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dictionaries

    A dictionary maps **keys** to **values**. Where a list answers "what is at position 3?",
    a dictionary answers "what is stored under `'age'`?" — which is usually the question you
    actually have.
    """)
    return


@app.cell
def _():
    subject = {"id": "sub-01", "age": 24, "handedness": "right"}

    print(subject["age"])
    print(subject.get("session", "not recorded"))  # a default instead of an error

    subject["session"] = 1  # add a new key
    for _key, _value in subject.items():
        print(f"  {_key:12} {_value}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tuples and sets

    A **tuple** is an ordered sequence like a list, but it cannot be changed after it is
    made. Use one when the fixedness is the point — coordinates, or a function returning
    several values at once.

    A **set** is an unordered collection with no duplicates, and it does the
    membership-and-overlap questions quickly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    left_set = mo.ui.multiselect(
        options=[str(i) for i in range(1, 11)],
        value=["1", "2", "3", "4", "5", "6"],
        label="set A:",
    )
    right_set = mo.ui.multiselect(
        options=[str(i) for i in range(1, 11)],
        value=["4", "5", "6", "7", "8"],
        label="set B:",
    )
    mo.hstack([left_set, right_set], justify="start", gap=2)
    return left_set, right_set


@app.cell(hide_code=True)
def _(left_set, mo, right_set):
    _A = {int(x) for x in left_set.value}
    _B = {int(x) for x in right_set.value}
    mo.md(f"""
    ```python
    A = {_A or set()}
    B = {_B or set()}

    A | B    # union         -> {_A | _B or set()}
    A & B    # intersection  -> {_A & _B or set()}
    A - B    # difference    -> {_A - _B or set()}
    A ^ B    # in one, not both -> {_A ^ _B or set()}
    ```
    """)
    return


@app.cell
def _():
    point = (3, 7)  # a tuple
    x, y = point  # unpacking: two names from one tuple
    print(f"x={x}, y={y}")

    try:
        point[0] = 99
    except TypeError as err:
        print(f"TypeError: {err}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modules

    Most of Python's usefulness lives in modules you import rather than in the language
    itself. The standard library ships with the interpreter; everything else you install.

    ```python
    import math                     # the whole module
    import numpy as np              # under a shorter name
    from math import sqrt, pi       # just the names you want
    ```

    Prefer `import numpy as np` to `from numpy import *`. With the second form you cannot
    tell where a name came from, and two modules can silently overwrite each other's.
    """)
    return


@app.cell
def _():
    import math
    import random

    random.seed(0)  # makes the "random" numbers reproducible

    print(f"sqrt(64)      = {math.sqrt(64)}")
    print(f"pi            = {math.pi:.5f}")
    print(f"random draws  = {[round(random.random(), 3) for _ in range(5)]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    Add a cell under each one and write your answer. Everything you need is above, and there
    is more than one right way to do each.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Find the even numbers

    Given `a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]`, make a new list containing only the
    even elements.

    *Hint: `%` from the arithmetic section.*

    ### 2. Find the range

    Given a list of integers with at least one element, return the difference between the
    largest and smallest values.

    *Hint: `max()` and `min()` are built in.*

    ### 3. Numbers in both lists

    Find the numbers that appear in **both** lists:

    ```python
    a = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361]
    b = [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]
    ```

    *Hint: a list comprehension with `in` works. So does one line using sets — try both and
    compare.*

    ### 4. Speeding ticket

    Write a function that takes a speed and returns the fine: `$0` at 60 or below, `$100`
    from 61 to 80 inclusive, `$500` at 81 or above.

    *You have seen this one already — write it yourself without scrolling up, then compare.*
    """)
    return


@app.cell
def _():
    # Exercise 1 -- the even numbers. Your answer below.
    ex1 = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

    ex1
    return


@app.cell
def _():
    # Exercise 2 -- largest minus smallest. Your answer below.
    ex2 = [17, 4, 9, 42, 25, 3]

    ex2
    return


@app.cell
def _():
    # Exercise 3 -- the numbers in both lists. Your answer below.
    ex3_a = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361]
    ex3_b = [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]

    len(ex3_a), len(ex3_b)
    return


@app.cell
def _():
    # Exercise 4 -- write the function, then try it on a few speeds.
    def my_fine(speed):
        return None

    [(s, my_fine(s)) for s in (55, 70, 95)]
    return


if __name__ == "__main__":
    app.run()
