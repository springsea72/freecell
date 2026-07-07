import unittest

from game_model import FreeCellGame
import learned_policy


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
class LearnedPolicyTests(unittest.TestCase):
    def test_choose_action_returns_current_legal_move(self):
        game = FreeCellGame(seed=1)
        bundle = learned_policy.create_model_bundle(seed=123, device="cpu")

        action = learned_policy.choose_action(game, bundle, device="cpu")

        self.assertIn(action, game.generate_legal_moves())

    def test_scoring_and_choose_action_do_not_mutate_game(self):
        game = FreeCellGame(seed=1)
        bundle = learned_policy.create_model_bundle(seed=123, device="cpu")
        before = game.state_key()

        scores = learned_policy.score_legal_moves(game, bundle, device="cpu")
        action = learned_policy.choose_action(game, bundle, device="cpu")

        self.assertGreater(len(scores), 0)
        self.assertIn(action, [move for move, _ in scores])
        self.assertEqual(before, game.state_key())


if __name__ == "__main__":
    unittest.main()
