"""Parameter bag for the MCTS-AHD search.

Kept as the upstream release has it, with the paper's defaults, plus the five
MCTS knobs that upstream hardcodes inside ``source/mcts.py`` and
``source/mcts_ahd.py``. Exposing them changes no default; it only makes the
paper's settings visible in ``meta.json`` and overridable from the command
line, which is what makes the ablations in Table 5 of the paper reproducible
here.
"""


class Paras:
    def __init__(self):
        # ── general ───────────────────────────────────────────────────────────
        self.method = 'mcts_ahd'
        self.problem = 'atsp_gls'
        self.management = 'pop_greedy'
        self.selection = 'prob_rank'

        # ── evolutionary settings (paper defaults) ────────────────────────────
        self.pop_size = 10          # |E|, the elite set                default 10
        self.init_size = 4          # N_I, initial tree nodes           default 4
        self.ec_fe_max = 1000       # T, evaluation budget              default 1000
        self.ec_operators = ['e1', 'e2', 'm1', 'm2', 's1']
        self.ec_m = 5               # max parents sampled for e1
        # Expansion weights: k children each from m1 and m2, one each from e2
        # and s1, i.e. 2k+2 children per expansion (paper, Section 3.2).
        self.ec_operator_weights = [0, 1, 2, 2, 1]

        # ── MCTS settings (paper, Sections 3.2 and 3.3) ───────────────────────
        self.mcts_exploration_constant = 0.1   # lambda_0
        self.mcts_alpha = 0.5                  # progressive widening exponent
        self.mcts_max_depth = 10
        self.mcts_expansion_children = 2       # k in "2k+2 children"
        self.mcts_seed = 2024

        # ── LLM ───────────────────────────────────────────────────────────────
        self.llm_use_local = False
        self.llm_local_url = None
        self.llm_api_endpoint = "chat.openai.com"
        self.llm_api_key = "Not used"          # the client carries the key
        self.llm_model = None                  # an instantiated client object

        # ── experiment ────────────────────────────────────────────────────────
        self.exp_debug_mode = False
        self.exp_output_path = "./"
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = -1

        # ── evaluation ────────────────────────────────────────────────────────
        self.eva_timeout = 60
        self.eva_numba_decorator = False

    def set_paras(self, *args, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"unknown parameter {key!r}")
            setattr(self, key, value)


if __name__ == "__main__":
    paras = Paras()
    paras.set_paras(mcts_alpha=0.7, ec_fe_max=200)
    print(paras.mcts_alpha, paras.ec_fe_max)
