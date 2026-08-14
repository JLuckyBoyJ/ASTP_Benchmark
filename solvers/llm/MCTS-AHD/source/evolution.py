"""The six LLM actions of MCTS-AHD, plus the thought-alignment process.

The prompt text below is the paper's (Appendix E.1 and E.2) and is reproduced
verbatim — i1 to create a heuristic from scratch, m1/m2 to mutate mechanism or
parameters, e1/e2 to cross over, s1 to reason over an MCTS tree path, and a
second call after every generation that re-describes the code that was actually
produced (the *thought-alignment* process of Section 3.1, which exists because
asking for the description first lets the model describe something it then does
not implement).

The task-specific parts spliced into every prompt — what the problem is, the
function name, its inputs and outputs, the domain hints — come from
``problem_adapter.Prompts`` and therefore from ``prompts/atsp_*/``. That split
is what retargets the method to ATSP without touching the method.

Changes from the released file, all marked ``ATSP:``:

* the ``use_local_llm`` branch instantiated ``LocalLLM``, a name the release
  never imports — it raised ``NameError`` for anyone who set the flag. Local
  models are reached through ``utils/llm_client/`` instead, like every other
  provider, so the dead branch is gone.
* ``input = lambda: ...`` at module scope shadowed the builtin so that the
  debug-mode ``input()`` calls silently did nothing. Debug mode now logs.
* code and design-idea extraction is hardened. Upstream runs
  ``re.search(r"\\{(.*?)\\}", response).group(1)`` with no ``None`` check, so a
  response without braces raised ``AttributeError`` inside a bare
  ``except``-and-retry loop in ``evolution_interface.py`` — an infinite stream
  of paid LLM calls. It also matched code with ``re.findall(r"import.*return")``,
  which swallows the surrounding `````python`` fence that gpt-4o-mini emits most of
  the time, producing a ``SyntaxError`` and wasting the evaluation. Fenced
  blocks are now preferred, the old regex remains as the fallback, and failure
  raises instead of looping.
"""

import logging
import re

from .interface_LLM import InterfaceAPI as InterfaceLLM

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_BRACE = re.compile(r"\{(.*?)\}", re.DOTALL)


class Evolution:

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        self._use_local_llm = kwargs.get('use_local_llm', False)
        self._url = kwargs.get('url')

        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode

        # ATSP: one path for every provider; see the module docstring.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key,
                                          self.model_LLM, self.debug_mode)

    # ── thought alignment (paper, Section 3.1) ────────────────────────────────

    def get_prompt_post(self, code, algorithm):
        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    def get_prompt_refine(self, code, algorithm):
        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    # ── the six actions (paper, Figure 2 and Appendix E.1) ────────────────────

    def get_prompt_i1(self):
        prompt_content = self.prompt_task + "\n" + "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv + \
                         "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_s1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    # ── response parsing (ATSP: hardened, see the module docstring) ───────────

    def _extract_code(self, response: str):
        """Return the function source, or ``None``.

        Preference order: a fenced block that defines a function; then
        upstream's ``import...return`` / ``def...return`` regex, with the
        output names re-appended as upstream does.
        """
        for block in _FENCE.findall(response):
            if "def " in block:
                return self._ensure_imports(block.strip())

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)
        if len(code) == 0:
            return None
        # Upstream cuts the response at the bare word `return` and re-appends
        # the declared output names.
        return self._ensure_imports(
            code[0] + " " + ", ".join(s for s in self.prompt_func_outputs))

    @staticmethod
    def _ensure_imports(code: str) -> str:
        if "np." in code and "import numpy" not in code:
            code = "import numpy as np\n" + code
        return code

    def _extract_thought(self, response: str, code: str | None) -> str:
        """The design idea: the braced sentence, if it precedes the code."""
        match = _BRACE.search(response)
        if match is not None and match.group(1).strip():
            code_start = min((idx for idx in (response.find("```"),
                                              response.find("\ndef "),
                                              response.find("\nimport "))
                              if idx != -1), default=len(response))
            if match.start() < code_start:
                return match.group(1).strip()

        # No usable brace: take whatever prose comes before the code.
        for marker in ("```", "\nimport ", "\ndef "):
            idx = response.find(marker)
            if idx > 0:
                text = response[:idx].strip()
                if text:
                    return text
        return (response.strip() or "no description provided")[:600]

    def _get_thought(self, prompt_content):
        return self.interface_llm.get_response(prompt_content, 0)

    def _get_alg(self, prompt_content, n_retry: int = 3):
        last_response = ""
        for attempt in range(1, n_retry + 1):
            response = self.interface_llm.get_response(prompt_content)
            last_response = response or ""
            code = self._extract_code(last_response)
            if code:
                return [code, self._extract_thought(last_response, code)]
            if self.debug_mode:
                logger.info("attempt %d: no code found in the response, retrying", attempt)
        raise ValueError(
            "the model returned no usable function after "
            f"{n_retry} attempts; last response began: {last_response[:200]!r}")

    def post_thought(self, code, algorithm):
        """The paper's thought-alignment call: describe the code as written."""
        return self._get_thought(self.get_prompt_refine(code, algorithm))

    # ── action entry points ───────────────────────────────────────────────────

    def _run(self, action: str, prompt_content):
        if self.debug_mode:
            logger.info("prompt for action [%s]:\n%s", action, prompt_content)
        code_all, algorithm = self._get_alg(prompt_content)
        if self.debug_mode:
            logger.info("action [%s] design idea: %s", action, algorithm)
            logger.info("action [%s] code:\n%s", action, code_all)
        return [code_all, algorithm]

    def i1(self):
        return self._run("i1", self.get_prompt_i1())

    def e1(self, parents):
        return self._run("e1", self.get_prompt_e1(parents))

    def e2(self, parents):
        return self._run("e2", self.get_prompt_e2(parents))

    def m1(self, parents):
        return self._run("m1", self.get_prompt_m1(parents))

    def m2(self, parents):
        return self._run("m2", self.get_prompt_m2(parents))

    def s1(self, parents):
        return self._run("s1", self.get_prompt_s1(parents))
