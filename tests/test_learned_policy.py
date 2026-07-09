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
    def test_model_forward_outputs_action_and_progress_scores(self):
        model = learned_policy.create_model(hidden_size=16, torch=torch)
        input_size = learned_policy.STATE_FEATURE_SIZE + learned_policy.MOVE_FEATURE_SIZE
        outputs = model(torch.zeros((3, input_size), dtype=torch.float32))

        self.assertEqual((3, 2), tuple(outputs.shape))
        self.assertEqual((3,), tuple(learned_policy.action_scores_from_output(outputs).shape))
        self.assertEqual((3,), tuple(learned_policy.progress_scores_from_output(outputs).shape))

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

    def test_score_legal_moves_uses_action_score_only(self):
        class ActionProgressModel:
            def to(self, device):
                return self

            def eval(self):
                return None

            def __call__(self, model_input):
                count = model_input.shape[0]
                action_scores = torch.arange(count, dtype=torch.float32, device=model_input.device)
                progress_scores = torch.arange(count, 0, -1, dtype=torch.float32, device=model_input.device) + 100.0
                return torch.stack([action_scores, progress_scores], dim=1)

        game = FreeCellGame(seed=1)
        bundle = learned_policy.ModelBundle(
            model=ActionProgressModel(),
            metadata=learned_policy.base_metadata(),
            device=torch.device("cpu"),
        )

        scored_moves = learned_policy.score_legal_moves(game, bundle, device="cpu")
        scores = [score for _, score in scored_moves]

        self.assertEqual([float(index) for index in range(len(scored_moves))], scores)
        self.assertEqual(scored_moves[-1][0], learned_policy.choose_action(game, bundle, device="cpu"))


if __name__ == "__main__":
    unittest.main()
