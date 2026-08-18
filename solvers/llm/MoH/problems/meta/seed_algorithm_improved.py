"""The optimizer that *drives* the outer loop on iteration 0.

Two distinct things are called "the seed" in MoH and it is worth keeping them
apart:

* ``seed_algorithm.py`` is ``I_0``, the optimizer whose meta-utility is measured
  before the loop starts. It is the baseline the run is judged against.
* this file is the function that performs the first outer-loop step — the thing
  that *writes* candidate optimizers, before MoH has discovered one of its own
  to take over the job (``run_meta_optimizer``: from iteration 1 onward the
  best-scoring candidate replaces it, which is the self-invocation in Section
  3.3).

It is a slightly richer strategy than ``I_0``: it samples at two temperatures,
caches utilities so an identical candidate is never scored twice, and keeps the
top ``batch_size`` of each round. The caching matters more here than anywhere
else — a duplicate candidate at this level costs N full subtask evaluations.
"""

from utils.utils import extract_code, extract_idea
import json


def improve_algorithm(population, utility, language_model, function_format, subtask):
    """Improve a solution according to a utility function."""
    expertise = ("You are an expert in the domain of designing meta optimization "
                 "strategy and designing combinatorial optimization problems. Your task "
                 "is to design heuristics that can effectively solve optimization "
                 "problems.")
    n_messages = getattr(language_model, "batch_size", 5)
    temperature_values = [0.7, 1.0]
    solutions_cache = set()
    new_solutions = []
    new_ideas = []
    utility_cache = {}

    def evaluate_solution(solution, idea):
        if solution not in utility_cache:
            utility_cache[solution] = utility(solution, idea, subtask)
        return utility_cache[solution]

    for temp in temperature_values:
        selected_solution = population.get_random_solution(subtask)
        direction_prompt = (
            f"Given the following heuristic for subtask: {selected_solution['best_sol']} "
            f"with its idea: {selected_solution['idea']}, and utility score: "
            f"{selected_solution['utility']}, summarize the key idea from this "
            "heuristic, then provide several totally different ideas from the given one "
            "to design improved algorithms. Propose some conventional optimization "
            "techniques, or some novel metaheuristic strategies that can be used to "
            "improve the current solution. For example, Population-Based, Heuristic "
            "(Partial) Search Methods, Local Search and Iterative Improvement, "
            "Bandit-Based (Exploration-Exploitation) Methods, etc. Please provide a "
            "single string as the answer, less than 50 words. Your response should be "
            "formatted as a json structure: "
            '```json\n{"insights":["content","content","content"]}\n```.'
        )
        response = language_model.prompt(expertise, direction_prompt)
        try:
            directions = json.loads(extract_code(response))["insights"]
        except Exception:
            continue

        message_batch = []
        for direction in directions:
            message_batch.append(f"""Improve the following solution:
```python
{selected_solution['best_sol']}
```
You must return an improved solution. Formatted as follows:
{function_format}
To better solve the problem, you are encouraged to develop new solutions based on the \
direction proposed: {direction}. If you think the direction is a refinement to the \
current solution, just improve the current solution. If you think the direction is a new \
idea, you can develop a brand new solution with NO RELATION to the solution given. You \
can add appropriate loops in the code for your needs, but not larger than 5.
You will be evaluated based on a score function. The lower the score, the better the \
solution. Be as creative as you can under the constraints.
Generate a solution with temperature={temp} that focuses on different aspects of \
optimization.""")

        responses = language_model.prompt_batch(expertise, message_batch, temperature=temp)
        generated_solutions = extract_code(responses)
        generated_ideas = extract_idea(responses)

        scored_solutions = [(idea, sol, evaluate_solution(sol, idea))
                            for sol, idea in zip(generated_solutions, generated_ideas)
                            if sol and sol not in solutions_cache]
        # Lowest utility first: the best candidates are the ones kept, not the
        # worst. Upstream sorted with reverse=True here and then took the head,
        # which kept whichever candidates scored *worst* in each round.
        scored_solutions.sort(key=lambda x: x[2])

        for idea, sol, _ in scored_solutions[:n_messages]:
            new_ideas.append(idea)
            new_solutions.append(sol)
            solutions_cache.add(sol)

    if not new_solutions:
        best = population.get_solution_by_index(subtask, 0)
        return best['idea'], best['best_sol'], best['utility']

    scored = [(idea, solution, evaluate_solution(solution, idea))
              for idea, solution in zip(new_ideas, new_solutions)]
    return min(scored, key=lambda x: x[2])
