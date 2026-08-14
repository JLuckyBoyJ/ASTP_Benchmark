import inspect
import logging
import os
import re

import hydra


def init_client(cfg):
    """Instantiate the LLM client named by the Hydra config.

    One path only: ``cfg.llm_client`` is a structured config under
    ``configs/llm/MCTS-AHD/cfg/llm_client/``, and Hydra instantiates whatever
    ``_target_`` it names. Upstream also accepted a bare ``cfg.model`` string
    and branched on its prefix to pick between three hardcoded providers; that
    shorthand is gone, because it silently ignored the timeout and retry
    settings the structured configs carry, and because
    ``llm_client=litellm llm_client.model=<provider>/<model>`` reaches every
    provider the prefix-matching ever did and several hundred it did not.

        python main.py llm_client=openai llm_client.model=gpt-4o
        python main.py llm_client=litellm llm_client.model=anthropic/claude-sonnet-4-5
        python main.py llm_client=stub
    """
    if cfg.get("model", None):
        raise ValueError(
            "`model=...` is upstream's shorthand and is not supported here; "
            "use `llm_client.model=...` (and `llm_client=<provider>` for a "
            "non-OpenAI endpoint) so the request timeout and retry settings in "
            "configs/llm/MCTS-AHD/cfg/llm_client/ still apply.")
    return hydra.utils.instantiate(cfg.llm_client)


def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def print_hyperlink(path, text=None):
    """Terminal hyperlink to a file or folder, for convenient navigation."""
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # no traceback found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1,
                        poll_seconds: float = 0.05, timeout: float = 60.0):
    """Wait until the evaluation subprocess has written its first output.

    Upstream spins in a tight ``while True`` re-reading the file with no sleep
    and no timeout: it burns a core while the child starts up, and it hangs
    forever if the child dies before writing anything (a missing interpreter,
    an import error at module scope with buffered stderr). Both are fixed here.
    """
    import time

    deadline = time.time() + timeout
    while True:
        try:
            log = file_to_string(stdout_filepath)
        except OSError:
            log = ""
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info("Iteration %s: code run %s raised (see %s)",
                             iter_num, response_id,
                             print_hyperlink(stdout_filepath, "stdout"))
            else:
                logging.info("Iteration %s: code run %s started", iter_num, response_id)
            return True
        if time.time() > deadline:
            logging.warning("Iteration %s: code run %s produced no output in %.0fs",
                            iter_num, response_id, timeout)
            return False
        time.sleep(poll_seconds)


def extract_description(response: str) -> str | None:
    """Code description enclosed between ``<start>`` and ``<end>`` / a fence."""
    for pattern in (r"<start>(.*?)```python", r"<start>(.*?)<end>"):
        match = re.search(pattern, response, re.DOTALL)
        if match is not None:
            return match.group(1).strip()
    return None


def extract_code_from_generator(content):
    """Extract code from an LLM response."""
    pattern_code = r"```python(.*?)```"
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        lines = content.split("\n")
        start = end = None
        for i, line in enumerate(lines):
            if line.startswith("def"):
                start = i
            if "return" in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = "\n".join(lines[start:end + 1])

    if code_string is None:
        return None
    if "np" in code_string and "import numpy" not in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string and "import torch" not in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove signature and import lines."""
    lines = code_string.split("\n")
    filtered_lines = []
    for line in lines:
        if line.startswith(("def", "import", "from")):
            continue
        if line.startswith("return"):
            filtered_lines.append(line)
            break
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name) and inspect.isfunction(getattr(module, func_name)):
            return func_name
    return None
