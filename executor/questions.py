"""Track interactive answers and replace only the current questions on screen."""

from contextvars import ContextVar
import os
import shutil
import sys
from unicodedata import category, combining, east_asian_width

from runner import ANSI, styled


active_history = ContextVar("active_question_history", default=None)


class PreviousQuestion(Exception):
    """Return to the preceding interactive prompt."""


def cursor_after(text, width, row=0, column=0):
    for char in ANSI.sub("", text):
        if char == "\n":
            row += 1
            column = 0
        elif char == "\r":
            column = 0
        elif char == "\t":
            column = min(width - 1, (column // 8 + 1) * 8)
        elif not combining(char) and category(char) not in {"Cc", "Cf"}:
            cells = min(width, 2 if east_asian_width(char) in "WF" else 1)
            if column + cells > width:
                row += 1
                column = 0
            column += cells
    return row, column


class QuestionDisplay:
    def __init__(self, stream, section):
        self.stream = stream
        self.section = section
        self.terminal = stream.isatty() and sys.stdin.isatty() and os.environ.get("TERM") != "dumb"
        self.size = shutil.get_terminal_size()
        self.resized = False
        self.transcript = ""
        self.rows = 0
        self.column = 0
        self.boundary = 0
        self.prompt_start = 0
        self.pending_error = None
        self.pending_notice = None

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()

    @property
    def position(self):
        return len(self.transcript)

    def _record(self, text):
        self.resized = self.resized or shutil.get_terminal_size() != self.size
        self.transcript += text
        self.rows, self.column = cursor_after(text, self.size.columns, self.rows, self.column)

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
        self.prompt_start = self.position
        error, self.pending_error = self.pending_error, None
        if error:
            print(error)
        print(prompt, end="", flush=True)
        value = input()
        self._record(value + "\n")
        if error:
            self.clear(self.prompt_start)
            print(prompt + value, flush=True)
        return value

    def mark(self):
        self.boundary = self.position

    def clear(self, start=None):
        if not self.terminal:
            return
        start = self.boundary if start is None else start
        prefix = self.transcript[:start]
        size = shutil.get_terminal_size()
        rows, column = cursor_after(prefix, size.columns)
        distance = self.rows - rows
        if self.resized or size != self.size or distance >= size.lines:
            # Relative cursor movement cannot reach a question that has scrolled off screen.
            self.stream.write("\x1b[2J\x1b[H" + prefix)
        elif distance or self.column != column or self.transcript != prefix:
            up = f"\x1b[{distance}F" if distance else "\r"
            horizontal = f"\x1b[{column + 1}G" if column else ""
            self.stream.write(up + horizontal + "\x1b[J")
        self.stream.flush()
        self.size = size
        self.resized = False
        self.transcript = prefix
        self.rows, self.column = rows, column

    def reject(self, message):
        self.clear(self.prompt_start)
        self.pending_error = message

    def show(self, job_number=None):
        self.pending_error = None
        self.clear()
        if self.pending_notice:
            print(self.pending_notice, flush=True)
            self.pending_notice = None
        self.section("Questions")
        if job_number is not None:
            print(styled(f"[Job {job_number}]", "1;36"), flush=True)

    def notice(self, message):
        self.pending_notice = message


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
