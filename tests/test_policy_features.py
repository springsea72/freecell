import unittest

from dataset_builder import sample_from_state
from game_model import Card, FreeCellGame, Move, MoveType, Suit
from policy_features import (
    MOVE_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    features_from_sample,
    validate_sample,
)


def card(suit, value):
    return Card(suit, value)


def empty_deal():
    return [[] for _ in range(8)]


def valid_sample():
    game = FreeCellGame(deal=empty_deal())
    game.columns[0] = [card(Suit.SPADES, 1)]
    return sample_from_state(
        game,
        Move(MoveType.COL_TO_HOME, 0),
        seed=1,
        source_trace="trace.json",
        step_index=0,
        remaining_moves=1,
    )


class PolicyFeatureTests(unittest.TestCase):
    def test_feature_dimensions_are_stable(self):
        state_features, move_features, action_index = features_from_sample(valid_sample())

        self.assertEqual(STATE_FEATURE_SIZE, len(state_features))
        self.assertGreater(len(move_features), 0)
        self.assertTrue(all(len(features) == MOVE_FEATURE_SIZE for features in move_features))
        self.assertIsInstance(action_index, int)

    def test_validate_sample_rejects_bad_action_index(self):
        sample = valid_sample()
        sample["action_index"] = len(sample["legal_moves"])

        with self.assertRaisesRegex(ValueError, "action_index"):
            validate_sample(sample)

    def test_validate_sample_rejects_bad_action(self):
        sample = valid_sample()
        sample["action"] = dict(sample["legal_moves"][0])
        sample["action"]["from_idx"] = 7

        with self.assertRaisesRegex(ValueError, "action"):
            validate_sample(sample)

    def test_validate_sample_rejects_empty_legal_moves(self):
        sample = valid_sample()
        sample["legal_moves"] = []
        sample["action_index"] = 0

        with self.assertRaisesRegex(ValueError, "no legal moves"):
            validate_sample(sample)


if __name__ == "__main__":
    unittest.main()
