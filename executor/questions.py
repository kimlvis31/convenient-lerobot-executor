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
        if self.terminal:
            self._record(text)
        return self.stream.write(text)

    def read(self, prompt):
        if not self.terminal:
            return input(prompt)
        print(prompt, end="", flush=True)
        value = input()
        self._record(value + "\n")
        return value

    def clear(self):
        if self.terminal and (self.rows or self.column):
            up = f"\x1b[{self.rows}F" if self.rows else "\r"
            self.stream.write(up + "\x1b[J")
            self.stream.flush()
        self.rows = 0
        self.column = 0

    def show(self, job_number=None):
        self.clear()
        self.section("Questions")
        if job_number is not None:
            print(styled(f"[Job {job_number}]", "1;36"), flush=True)

    def notice(self, message):
        self.clear()
        print(message, file=self.stream, flush=True)


class QuestionHistory:
    def __init__(self, display):
        self.display = display
        self.answers = []
        self.position = 0

    def run(self, configure, job_number=None):
        token = active_history.set(self)
        try:
            while True:
                self.position = 0
                self.display.show(job_number)
                try:
                    return configure()
                except PreviousQuestion:
                    if not self.answers:
                        raise
                    self.rewind()
        finally:
            active_history.reset(token)

    def rewind(self):
        self.position -= 1
        del self.answers[self.position:]


def retry_question():
    history = active_history.get()
    if history is not None:
        history.rewind()
