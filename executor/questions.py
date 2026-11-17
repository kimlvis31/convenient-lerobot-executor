"""Track interactive answers and replace only the current questions on screen."""

from contextvars import ContextVar
import os
import shutil
import sys
from unicodedata import combining, east_asian_width

from runner import ANSI, styled


active_history = ContextVar("active_question_history", default=None)


class PreviousQuestion(Exception):
    """Return to the preceding interactive prompt."""


class QuestionDisplay:
    def __init__(self, stream, section):
        self.stream = stream
        self.section = section
        self.terminal = stream.isatty() and sys.stdin.isatty() and os.environ.get("TERM") != "dumb"
        self.rows = 0
        self.column = 0
        self.boundary = 0
        self.prompt_start = 0
        self.retrying = False

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def _record(self, text):
        width = shutil.get_terminal_size().columns
        for char in ANSI.sub("", text):
            if char == "\n":
                self.rows += 1
                self.column = 0
            elif char == "\r":
                self.column = 0
            elif not combining(char):
                cells = 2 if east_asian_width(char) in "WF" else 1
                if self.column + cells > width:
                    self.rows += 1
                    self.column = 0
                self.column += cells

    def write(self, text):
        history = active_history.get()
        if history is not None and history.replaying:
            return len(text)
        if self.terminal:
            self._record(text)
        return self.stream.write(text)

    def read(self, prompt):
        if not self.terminal:
            return input(prompt)
        if not self.retrying:
            self.prompt_start = self.rows
        self.retrying = False
        print(prompt, end="", flush=True)
        value = input()
        self._record(value + "\n")
        return value

    def mark(self):
        self.boundary = self.rows

    def clear(self, start=None):
        start = self.boundary if start is None else start
        distance = self.rows - start
        if self.terminal and (distance or self.column):
            up = f"\x1b[{distance}F" if distance else "\r"
            self.stream.write(up + "\x1b[J")
            self.stream.flush()
        self.rows = start
        self.column = 0

    def reject(self, message):
        self.clear(start=self.prompt_start)
        print(message, flush=True)
        self.retrying = True

    def show(self, job_number=None):
        self.retrying = False
        self.clear()
        self.section("Questions")
        if job_number is not None:
            print(styled(f"[Job {job_number}]", "1;36"), flush=True)

    def notice(self, message):
        self.clear()
        print(message, file=self, flush=True)
        self.mark()


class QuestionHistory:
    def __init__(self, display):
        self.display = display
        self.answers = []
        self.marks = []
        self.position = 0
        self.replaying = False

    def run(self, configure, job_number=None):
        token = active_history.set(self)
        try:
            self.replaying = False
            self.display.show(job_number)
            while True:
                self.position = 0
                try:
                    return configure()
                except PreviousQuestion:
                    if not self.answers:
                        raise
                    self.display.clear(self.rewind())
                    self.replaying = True
        finally:
            active_history.reset(token)

    def rewind(self):
        self.position -= 1
        mark = self.marks[self.position]
        del self.answers[self.position:]
        del self.marks[self.position:]
        return mark


def retry_question(message, *, accepted=True):
    history = active_history.get()
    if accepted and history is not None:
        history.rewind()
    if isinstance(sys.stdout, QuestionDisplay) and sys.stdout.terminal:
        sys.stdout.reject(message)
    else:
        print(message)
