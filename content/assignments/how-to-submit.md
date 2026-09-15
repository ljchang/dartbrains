# How to do an assignment

Assignments are marimo notebooks that check your work as you go and let you submit each question from inside the notebook. This page walks through the whole loop once; after that it is the same for every assignment.

## 1. Open the assignment

Each assignment belongs to a chapter and opens inside it. Go to the chapter — the [Programming](../Introduction_to_Programming/) chapter, say — and press **Assignment** in the header, or **Open assignment** on the card at the end of the page. The assignment opens in a drawer along the bottom of the window and starts running in your browser; the first time, it spends a few seconds installing its packages. The chapter stays where it is above the drawer, so you can read it while you work, and **Minimize** tucks the drawer down to its bar without stopping the notebook.

**Your work is saved in this browser as you type.** The drawer is your own copy of the assignment: close the tab, come back tomorrow, and your answers are where you left them. **History** in the drawer's bar shows every version of that copy — checkpoints as you work, a version each time you submit, and any you name yourself — and lets you look at one, compare it with your current copy, and restore it. Because the copy lives in this browser, work on one machine, in one browser, and press **Download .py** if you want a backup of your own.

If you would rather work outside the browser, press **Download .py** in the drawer and run:

```bash
uv run marimo edit --sandbox glm.py
```

Everything below works the same way there.

## 2. Work and check

Each question has a place for your code or your written answer, followed by a **Check** cell. Check runs instantly, as often as you like, and tells you whether your answer passes the visible tests. A cell that says *Complete the code cell above first* is simply waiting for your answer.

Checks are for learning. They are not your grade, and some questions run extra hidden tests when you submit.

## 3. Sign in and submit

Press **Grader · sign in** in the drawer's bar, or **Sign in with Dartmouth** near the top of the notebook. A tab opens on Dartmouth's login page (with Duo); when it says you can close the tab, the notebook shows you as signed in. Your password never enters the notebook. One sign-in covers every assignment on this site and lasts eight hours.

Then press **Submit** under a question. Autograded questions come back with a score and feedback within about a minute; written questions show *waiting for grade* until an instructor reads them. You can submit again; the assignment says which attempt counts. Each submission is also kept in **History**, so you can always see what you sent.

## 4. See your feedback

Under each question there is a **Feedback** cell. It loads on its own when you open the notebook and shows your latest attempt for that question: the score, the maximum, any comments, and how many attempts you have made. **Refresh** checks again without re-running the cell.

Autograded questions fill it in within about a minute of submitting. Written questions show your attempt until an instructor reads it, then the comments appear there too — so come back to this cell rather than waiting for an email.

Your sign-in lasts eight hours. Coming back the next day, the Feedback cell will say *Sign in to see your feedback* with a button right there; one click and your score appears. That is normal, not an error.

Everything is also on the grader's website at [grader.dartbrains.org](https://grader.dartbrains.org), where you can see every assignment and every attempt after signing in.

## If the assignment is updated

When we publish a new version of an assignment, the drawer's bar says *update available* and offers **Update…**. Updating replaces your copy with the new version and keeps the old one in **History**, so nothing is lost either way — and you can undo it right after. Your submitted attempts are on the grader and are not affected.

## If something looks wrong

- A red traceback in a Check cell after you have written an answer is a Python error in your code; read its last line.
- *Could not reach the grader* means your network is blocking the grading server; try another network or submit later.
- Nothing appears when you press **Assignment**: the assignment needs a laptop or desktop browser — on a phone the chapter is readable but the notebook does not run.
- Your answers are gone after clearing your browser's site data or working in a private window; that is where the copy lives. Use **Download .py** to keep a backup.
