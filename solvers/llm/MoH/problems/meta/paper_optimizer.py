"""A meta-optimizer MoH discovered on TSP, kept as an alternative starting point.

This is the ``improve_algorithm`` reported in the upstream release — an
ACO-flavoured strategy that maintains pheromone levels over *insights* rather
than over solutions, reinforcing the natural-language directions that keep
producing good code. It is one of the examples Appendix E of the paper uses to
argue that the outer loop invents recognisable metaheuristics (ACO in Fig. 10,
PSO in Fig. 11, simulated annealing in Fig. 12, tabu search in Fig. 13) rather
than shuffling prompts.

It was discovered on **symmetric TSP**, not on ATSP, so it is not a result here.
It is useful in two ways:

* as a stronger ``I_0`` than ``seed_algorithm.py``, to see whether an optimizer
  transfers across problems::

      python main.py problem=atsp_gls meta_optimizer=problems/meta/paper_optimizer.py

* as a worked example of what a *discovered* optimizer looks like, next to the
  hand-written seed, when reading ``code/improver/`` from a finished run.

Nothing in it references TSP, which is the point the paper makes about
cross-problem transfer in Appendix A ("Generalizability Claim"): the optimizer
operates on populations and prompts, so the downstream problem only enters
through ``function_format`` and ``utility``.
"""

from utils.utils import extract_code, extract_idea
import json


# {This metaheuristic employs an adaptive exploration-exploitation strategy that
# combines real-time performance evaluation of solutions with dynamic exploration
# rates, in a genetic-algorithm framework integrated with adaptive tabu-like
# mechanisms for efficient solution refinement.}
def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in optimizing metaheuristic strategies and combinatorial "
        "optimization problems. Your task is to design effective heuristics to solve "
        "optimization challenges."
    )

    elite_count = 4
    diversity_count = 3
    pheromone_levels = {}

    # Step 1: elite and diverse solutions
    population_size = population.get_subtask_size(task)
    elite_solutions = [population.get_solution_by_index(task, i)
                       for i in range(min(elite_count, population_size))]
    diverse_solutions = [population.get_random_solution(task)
                         for _ in range(diversity_count)]
    selected_solutions = elite_solutions + diverse_solutions

    # Step 2: gather insights, weighting each by how often it is proposed
    for solution in selected_solutions:
        temperature = 1 if solution['utility'] > 0 else 0.75
        prompt = (
            f"Given the solution '{solution['best_sol']}' with utility score "
            f"'{solution['utility']}', please suggest innovative optimization strategies "
            "that could enhance this code. Return your recommendations in JSON format: "
            '```json {"insights":["content","content"]} ```.'
        )
        response = language_model.prompt(expertise, prompt, temperature=temperature)
        try:
            insights = json.loads(extract_code(response))["insights"]
            for insight in insights:
                pheromone_levels[insight] = pheromone_levels.get(insight, 1.0) + 1.0
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # Step 3: rank directions by pheromone level
    sorted_insights = sorted(pheromone_levels.items(), key=lambda x: x[1], reverse=True)
    top_insights = [insight for insight, level in sorted_insights if level > 1.0]

    # Step 4: one message per surviving direction
    message_batch = []
    for direction in top_insights:
        message_batch.append(
            f"Refine the solution for the task '{task}' by focusing on this optimization "
            f"approach: {direction}. "
            f"Consider elite solutions: {[sol['best_sol'] for sol in elite_solutions]}. "
            f"Ensure your output adheres to the following format: {function_format}. "
            "In addition, provide a summary of changes made."
        )
    if not message_batch:
        best_existing = population.get_solution_by_index(task, 0)
        return best_existing['idea'], best_existing['best_sol'], best_existing['utility']

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)

    # Step 5: evaluate; reinforce the insights that produced good code
    solutions_with_utilities = []
    for response in responses:
        try:
            new_solution = extract_code(response)
            important_idea = extract_idea(response)
            if not new_solution:
                continue
            score = utility(new_solution, important_idea, task)
            pheromone_levels[important_idea] = (
                pheromone_levels.get(important_idea, 1.0) + (2.0 / (score + 1e-6)))
            solutions_with_utilities.append((important_idea, new_solution, score))
        except Exception:
            continue

    if not solutions_with_utilities:
        best_existing = population.get_solution_by_index(task, 0)
        return best_existing['idea'], best_existing['best_sol'], best_existing['utility']

    return min(solutions_with_utilities, key=lambda x: x[2])
