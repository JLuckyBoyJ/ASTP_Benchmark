"""Text wrangling shared by MoH's two loops.

Upstream MoH's helpers, kept because the *generated* meta-optimizers call them
by name — ``extract_code`` and ``extract_idea`` are named in the meta prompt and
appear at the top of every optimizer the LLM writes, so their signatures are
part of MoH's contract with the model and cannot be renamed.

Three additions, all of them fixes for things that silently cost a run:

* :func:`ensure_numpy_import` — the LLM routinely writes ``np.`` without the
  import, because the prompt shows it a numpy signature. Upstream then imports
  ``gpt.py`` and dies on ``NameError``, which is scored 1e6 and read as "bad
  heuristic" rather than "missing import".
* :func:`extract_eval_output` — the parent process used to scan the *whole*
  evaluation log for the substring ``"Error"``, and the log begins with the
  heuristic's own source. A rule with the word ``error`` in a comment was
  discarded no matter how well it scored.
* :func:`parse_objective` — reads the last numeric line rather than the last
  line, so a trailing warning from numpy does not turn a good evaluation into
  a parse failure.
"""

import os
import re
import traceback


def code_only(code_string):
    """Remove # comments and redundant blank lines."""
    code_without_comments = re.sub(r"#.*", "", code_string)
    cleaned_code = re.sub(r"\n\s*\n", "\n", code_without_comments)
    return cleaned_code.strip()


def clean_code(algorithm_str):
    if isinstance(algorithm_str, str):
        return code_only(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [code_only(s) for s in algorithm_str]


def extract_idea(algorithm_str):
    if isinstance(algorithm_str, str):
        return find_braces(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [extract_idea(s) for s in algorithm_str]


def find_braces(response):
    """Extract content inside first {} pair."""
    match = re.search(r"\{(.*?)\}", response, re.DOTALL)
    if match:
        return match.group(1)
    if "import" in response:
        return response.split("import")[0]
    return None


def extract_code(algorithm_str):
    """Extract largest markdown code block."""
    if isinstance(algorithm_str, str):
        return find_largest_code_block_line_by_line(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [extract_code(s) for s in algorithm_str]


def find_largest_code_block_line_by_line(text):
    """Find the largest ``` code block in text."""
    largest_block = ""
    current_block = ""
    nesting_level = 0
    lines = text.split("\n")

    for line in lines:
        if line.startswith("```"):
            if not line[3:].strip():  # closing delimiter
                nesting_level -= 1
                if nesting_level == 0:
                    current_block += line + "\n"
                    if len(current_block) > len(largest_block):
                        largest_block = current_block
                    current_block = ""
                else:
                    current_block += line + "\n"
            else:  # opening delimiter
                current_block += line + "\n"
                nesting_level += 1
        else:
            if nesting_level > 0:
                current_block += line + "\n"

    if largest_block:
        largest_block = "\n".join(largest_block.strip().split("\n")[1:-1])

    return largest_block if largest_block else None


def find_txt_block(string):
    """Extract content of ```txt code block."""
    inside_txt_block = False
    current_block = ""
    lines = string.split("\n")

    for line in lines:
        if line.strip() == "```txt":
            inside_txt_block = True
            current_block = line + "\n"
        elif line.strip() == "```" and inside_txt_block:
            current_block += line + "\n"
            inside_txt_block = False
        elif inside_txt_block:
            current_block += line + "\n"
    return current_block


def match_number(string):
    """Extract first integer from string, default 1."""
    match = re.search(r"\d+", string)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return 1
    return 1


def read_file_as_str(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_str_to_file(s, path, mode="w"):
    if isinstance(s, list):
        s = "\n\n".join(s)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(s)
    except Exception as e:
        print("Failed to write to file", path, "with exception", e)
        print("Traceback:", traceback.format_exc())
        s = str(s)
        with open(path, mode, encoding="utf-8") as f:
            f.write(s)


# ── additions ────────────────────────────────────────────────────────────────

def ensure_numpy_import(code_string: str) -> str:
    """Prepend ``import numpy as np`` when the code uses ``np`` without it.

    The task prompts describe numpy signatures, so models write ``np.copy``
    freely and omit the import roughly a third of the time. Without this the
    heuristic raises ``NameError`` at import, scores 1e6, and is discarded as a
    bad idea rather than as a missing line.
    """
    if not code_string:
        return code_string
    if re.search(r"\bnp\s*\.", code_string) and not re.search(
            r"^\s*import\s+numpy\b", code_string, re.MULTILINE):
        return "import numpy as np\n" + code_string
    return code_string


#: The marker ``moh.py`` writes between the candidate's source and the
#: evaluation subprocess's stdout, so the two can be told apart afterwards.
EVAL_OUTPUT_MARKER = "# ===== Eval Output ====="


def extract_eval_output(stdout_str: str) -> str:
    """Return only the evaluation subprocess's output, not the code above it."""
    if EVAL_OUTPUT_MARKER in stdout_str:
        return stdout_str.split(EVAL_OUTPUT_MARKER, 1)[1]
    return stdout_str


def parse_objective(stdout_str: str) -> float | None:
    """The last line of evaluation output that parses as a float.

    ``eval.py`` prints the objective last, but numpy and the OpenMP runtime
    occasionally append a warning after it. Scanning backwards for the last
    numeric line is robust to that and identical otherwise.
    """
    for line in reversed(extract_eval_output(stdout_str).strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def print_hyperlink(path, text=None):
    """Terminal hyperlink to a file or folder, for convenient navigation."""
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"
