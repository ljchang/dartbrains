# How to do an assignment

Assignments are marimo notebooks that check your work as you go and let you submit each question from inside the notebook. This page walks through the whole loop once; after that it is the same for every assignment.

## 1. Open the assignment

Each assignment has a page in this section with a preview of the notebook. Press **Open in molab** to run it in [molab](https://molab.marimo.io), marimo's hosted notebook service. You need a molab account (GitHub or Google); it is separate from your Dartmouth login, which happens later inside the notebook. The first time you open an assignment molab installs its packages, which can take about a minute.

If you would rather work on your own computer, press **Download** and run:

```bash
uv run marimo edit --sandbox glm.py
```

## 2. Work and check

Each question has a place for your code or your written answer, followed by a **Check** cell. Check runs instantly, as often as you like, and tells you whether your answer passes the visible tests. A cell that says *Complete the code cell above first* is simply waiting for your answer.

Checks are for learning. They are not your grade, and some questions run extra hidden tests when you submit.

## 3. Sign in and submit

Press **Sign in with Dartmouth** near the top of the notebook. A tab opens on Dartmouth's login page (with Duo); when it says you can close the tab, the notebook shows you as signed in. Your password never enters the notebook.

Then press **Submit** under a question. Autograded questions come back with a score and feedback within about a minute; written questions show *waiting for grade* until an instructor reads them. You can submit again; the assignment page says which attempt counts.

## 4. See your feedback

Under each question there is a **Feedback** cell. It loads on its own when you open the notebook and shows your latest attempt for that question: the score, the maximum, any comments, and how many attempts you have made. **Refresh** checks again without re-running the cell.

Autograded questions fill it in within about a minute of submitting. Written questions show your attempt until an instructor reads it, then the comments appear there too — so come back to this cell rather than waiting for an email.

Your sign-in lasts eight hours. Coming back the next day, the Feedback cell will say *Sign in to see your feedback* with a button right there; one click and your score appears. That is normal, not an error.

Everything is also on the grader's website at [grader.dartbrains.org](https://grader.dartbrains.org), where you can see every assignment and every attempt after signing in.

## If something looks wrong

- A red traceback in a Check cell after you have written an answer is a Python error in your code; read its last line.
- *Could not reach the grader* means your network is blocking the grading server; try another network or submit later.
- If a *newer version* notice appears after you submit, the instructor updated the assignment. Your submission still counts; reopen the assignment page to get the current copy.
