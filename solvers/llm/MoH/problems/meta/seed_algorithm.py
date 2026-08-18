"""``I_0`` — the seed meta-optimizer, and the reference baseline of a run.

Figure 8 of the paper. Deliberately plain: pick one heuristic from the
population, ask the model for several *different* ideas, generate one candidate
per idea, keep the best. Everything MoH discovers in the outer loop is measured
against what this achieves, so it should stay simple — a clever seed would make
the improvement curve look flat for the wrong reason.

The signature is the one in Figure 3, shared by heuristic-optimizers and
meta-optimizers alike, which is what lets MoH invoke an optimizer on itself.
"""

from utils.utils import extract_code, extract_idea
import json


def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = ("You are an expert in the domain of designing meta optimization "
                 "strategy and combinatorial optimization problems. Your task is to "
                 "design heuristics that can effectively solve optimization problems.")

    # Step 1: select one solution from the population
    selected_solution = population.get_random_solution(task)

    # Step 2: ask for directions that differ from it
    direction_prompt = (
        f"Given the following heuristic for subtask: {selected_solution['best_sol']} "
        f"with its idea: {selected_solution['idea']} and utility score: "
        f"{selected_solution['utility']}, "
        "summarize the key idea from this heuristic, then provide several totally "
        "different ideas from the given one to design improved algorithms with a lower "
        "utility score. Provide a single string as the answer, less than 50 words. "
        "Your response should be formatted as a json structure: "
        '```json\n{"insights":["content","content","content"]}\n```.'
    )
    response = language_model.prompt(expertise, direction_prompt, temperature=1)
    try:
        directions = json.loads(extract_code(response))["insights"]
    except Exception:
        return selected_solution['idea'], selected_solution['best_sol'], \
            selected_solution['utility']

    # Step 3: one message per direction
    message_batch = []
    for direction in directions:
        message_batch.append(
            f"Improve the following solution:\n"
            f"```python\n{selected_solution['best_sol']}\n```\n"
            f"You must return an improved solution. Formatted as follows:\n"
            f"{function_format}\n"
            f"To better solve the problem, you are encouraged to develop new solutions "
            f"based on the direction proposed: {direction}. "
            "You will be evaluated based on a score function. The lower the score, the "
            "better the solution.\n"
            "Your response must firstly provide a summary of the key idea inside a "
            "brace and marked as a comment, followed by the code implementation. "
            "Be as creative as you can under the constraints."
        )

    # Step 4: generate
    responses = language_model.prompt_batch(expertise, message_batch, temperature=1)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    # Step 5: evaluate and keep the best
    scored = [(idea, solution, utility(solution, idea, task))
              for idea, solution in zip(new_ideas, new_solutions) if solution]
    if not scored:
        return selected_solution['idea'], selected_solution['best_sol'], \
            selected_solution['utility']
    return min(scored, key=lambda item: item[2])
