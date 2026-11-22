"""Jev-MDP: Jev's contract rebuilt as an MDP (research series v1-v10)."""
from .primitives import (Question, MultiChoiceQuestion, solve_mdp,
                         wilson_interval, ece, dict_to_text, apply_mask,
                         expected_cost_action)
from .engine import JevMDP, make_tickets, default_ticket_questions
from .multitask import MultiTaskScorer
from .matrices import (StateMatrix, ProbabilisticMatrix, RewardMatrix,
                       SuccessorFeatures, policy_evaluate_flow)
from .calibration import (AdaptiveConformal, SplitConformal, APSConformal,
                          ConformalRiskController, OODDetector, DriftDetector,
                          c2st, em_label_shift)
from .memory import KNNMemory
from .evaluation import (ope_report, ope_ensemble, fqe_value, mis_value,
                         hcope_lower_bound, PolicyVerifier, iqm_ci,
                         performance_profile)
from .information import evsi_query, gittins_index, sprt_decision, bwk_pacing
from .compile import compile_program, serve_compiled

__version__ = "10.1.0"
