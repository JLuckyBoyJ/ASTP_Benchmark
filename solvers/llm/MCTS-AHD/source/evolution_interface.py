"""Glue between an MCTS action and one evaluated offspring.

``get_algorithm`` produces the initial nodes; ``evolve_algorithm`` produces one
child during expansion. Both call an action in ``evolution.py``, run the
thought-alignment pass, evaluate the result on the training split and return the
offspring — or ``None`` when the budget for that expansion is spent.

Changes from the released file, all marked ``ATSP:``:

* ``get_offspring`` wrapped everything in ``while True: try/except: print(e)``.
  Any persistent failure — a bad API key, a prompt the model will not answer
  with code, a typo in a prompt file — became an unbounded loop of paid LLM
  calls with no way out but Ctrl-C. It now gives up after a few attempts.
* ``get_algorithm`` looped forever in the same way when every candidate
  returned ``inf`` or a duplicate objective. It now stops and reports, which is
  what lets ``mcts_ahd.run()`` raise a message pointing at
  ``evaluations/index.jsonl`` instead of hanging.
* failures are logged with their cause rather than printed bare.
"""

import copy
import logging
import random
import warnings

import numpy as np

from .evolution import Evolution

logger = logging.getLogger(__name__)

#: How many times one action may be re-issued when the model returns something
#: unusable (no code, or code identical to a heuristic already in the set).
MAX_GENERATION_ATTEMPTS = 3

#: How many candidates ``get_algorithm`` may burn before admitting the task is
#: broken. Initialisation is the only place this matters: expansion already has
#: its own three-attempt budget, inherited from the release.
MAX_INIT_ATTEMPTS = 12


class InterfaceEC:
    def __init__(self, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob,
                 select, n_p, timeout, use_numba, **kwargs):
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode, prompts,
                              **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

    # ── helpers ───────────────────────────────────────────────────────────────

    def code2file(self, code):
        with open("./ael_alg.py", "w", encoding="utf-8") as file:
            file.write(code)

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    logger.info("duplicated objective, retrying")
                return False
        population.append(offspring)
        return True

    def check_duplicate_obj(self, population, obj):
        return any(obj == ind['objective'] for ind in population)

    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)

    def population_generation_seed(self, seeds):
        population = []
        fitness = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])
        for i in range(len(seeds)):
            seed_alg = {
                'algorithm': seeds[i]['algorithm'],
                'code': seeds[i]['code'],
                'objective': None,
                'other_inf': None,
            }
            seed_alg['objective'] = np.round(np.array(fitness[i]), 5)
            population.append(seed_alg)
        logger.info("Initialisation finished: %d seed algorithms", len(seeds))
        return population

    # ── one offspring ─────────────────────────────────────────────────────────

    def _get_alg(self, pop, operator, father=None):
        offspring = {'algorithm': None, 'thought': None, 'code': None,
                     'objective': None, 'other_inf': None}
        if operator == "i1":
            parents = None
            offspring['code'], offspring['thought'] = self.evol.i1()
        elif operator == "e1":
            real_m = min(random.randint(2, self.m), len(pop))
            parents = self.select.parent_selection_e1(pop, real_m)
            offspring['code'], offspring['thought'] = self.evol.e1(parents)
        elif operator == "e2":
            other = copy.deepcopy(pop)
            if father in pop:
                other.remove(father)
            parents = self.select.parent_selection(other, 1)
            parents.append(father)
            offspring['code'], offspring['thought'] = self.evol.e2(parents)
        elif operator == "m1":
            parents = [father]
            offspring['code'], offspring['thought'] = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = [father]
            offspring['code'], offspring['thought'] = self.evol.m2(parents[0])
        elif operator == "s1":
            parents = pop
            offspring['code'], offspring['thought'] = self.evol.s1(pop)
        else:
            raise ValueError(f"evolution operator [{operator}] is not implemented")

        # The paper's thought-alignment pass: describe the code that was
        # actually produced, not the one the model said it would produce.
        offspring['algorithm'] = self.evol.post_thought(offspring['code'],
                                                        offspring['thought'])
        return parents, offspring

    def get_offspring(self, pop, operator, father=None):
        """One candidate from ``operator``, or ``(None, None)`` if it will not come."""
        last_error = None
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                parents, offspring = self._get_alg(pop, operator, father=father)
            except Exception as exc:      # ATSP: bounded, and the cause is kept
                last_error = exc
                logger.warning("action %s attempt %d/%d failed: %s",
                               operator, attempt, MAX_GENERATION_ATTEMPTS, exc)
                continue
            if not self.check_duplicate(pop, offspring['code']):
                return parents, offspring
            logger.info("action %s attempt %d/%d produced a duplicate heuristic",
                        operator, attempt, MAX_GENERATION_ATTEMPTS)
        if last_error is not None:
            logger.warning("action %s gave up after %d attempts: %s",
                           operator, MAX_GENERATION_ATTEMPTS, last_error)
        return None, None

    # ── initialisation and expansion ──────────────────────────────────────────

    def get_algorithm(self, eval_times, pop, operator):
        """An initial tree node. Returns ``(eval_times, pop, offspring|None)``."""
        for _ in range(MAX_INIT_ATTEMPTS):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator)
            if offspring is None:
                continue
            objs = self.interface_eval.batch_evaluate([offspring['code']], eval_times)
            if objs == 'timeout':
                logger.info("initial candidate timed out; trying another")
                continue
            if objs[0] == float('inf'):
                continue
            objective = np.round(objs[0], 5)
            if self.check_duplicate_obj(pop, objective):
                logger.info("initial candidate duplicates objective %s; trying another",
                            objective)
                continue
            offspring['objective'] = objective
            return eval_times, pop, offspring
        return eval_times, pop, None

    def evolve_algorithm(self, eval_times, pop, node, brother_node, operator):
        """One child of ``node``. Returns ``(eval_times, offspring|None)``."""
        for _ in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            if offspring is None:
                continue
            objs = self.interface_eval.batch_evaluate([offspring['code']], eval_times)
            if objs == 'timeout':
                return eval_times, None
            if objs[0] == float('inf') or self.check_duplicate(pop, offspring['code']):
                continue
            offspring['objective'] = np.round(objs[0], 5)
            return eval_times, offspring
        return eval_times, None
