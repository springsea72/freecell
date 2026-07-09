import unittest

from dataset_builder import sample_from_state, state_to_dict
from game_model import Card, FreeCellGame, Move, MoveType, Suit
from policy_features import (
    FEATURE_VERSION,
    MOVE_INCREASES_HOME_IDX,
    MOVE_OCCUPIES_FREE_IDX,
    MOVE_RELEASES_FREE_IDX,
    MOVE_RELEASES_LOW_IDX,
    MOVE_SOURCE_EMPTIED_IDX,
    MOVE_FEATURE_SIZE,
    MOVE_TO_EMPTY_COLUMN_IDX,
    STATE_BURIED_LOW_START,
    STATE_FEATURE_SIZE,
    STATE_HOME_NEXT_START,
    STATE_MOVABLE_SUFFIX_START,
    SUITS,
    features_from_sample,
    move_to_features,
    state_to_features,
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

        self.assertEqual(2, FEATURE_VERSION)
        self.assertEqual(272, STATE_FEATURE_SIZE)
        self.assertEqual(46, MOVE_FEATURE_SIZE)
        self.assertEqual(STATE_FEATURE_SIZE, len(state_features))
        self.assertGreater(len(move_features), 0)
        self.assertTrue(all(len(features) == MOVE_FEATURE_SIZE for features in move_features))
        self.assertIsInstance(action_index, int)

    def test_features_from_sample_accepts_dataset_schema_v1(self):
        sample = valid_sample()

        state_features, move_features, action_index = features_from_sample(sample)

        self.assertEqual(1, sample["version"])
        self.assertEqual(STATE_FEATURE_SIZE, len(state_features))
        self.assertTrue(move_features)
        self.assertEqual(sample["action_index"], action_index)

    def test_home_next_value_features(self):
        game = FreeCellGame(deal=empty_deal())
        game.home_cells[Suit.SPADES] = [card(Suit.SPADES, 1), card(Suit.SPADES, 2)]
        features = state_to_features(state_to_dict(game))

        spades_idx = SUITS.index("SPADES")
        hearts_idx = SUITS.index("HEARTS")
        self.assertAlmostEqual(3 / 13.0, features[STATE_HOME_NEXT_START + spades_idx])
        self.assertAlmostEqual(1 / 13.0, features[STATE_HOME_NEXT_START + hearts_idx])

    def test_column_suffix_and_buried_low_features(self):
        game = FreeCellGame(deal=empty_deal())
        game.columns[0] = [
            card(Suit.SPADES, 1),
            card(Suit.CLUBS, 7),
            card(Suit.HEARTS, 6),
            card(Suit.SPADES, 5),
        ]
        game.columns[1] = [card(Suit.HEARTS, 2)]
        features = state_to_features(state_to_dict(game))

        self.assertAlmostEqual(3 / 13.0, features[STATE_MOVABLE_SUFFIX_START])
        self.assertEqual(1.0, features[STATE_BURIED_LOW_START])
        self.assertEqual(0.0, features[STATE_BURIED_LOW_START + 1])

    def test_move_features_source_emptied_and_free_cell_occupied(self):
        game = FreeCellGame(deal=empty_deal())
        game.columns[0] = [card(Suit.SPADES, 5)]
        state = state_to_dict(game)

        features = move_to_features(
            state,
            {"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1},
        )

        self.assertEqual(1.0, features[MOVE_SOURCE_EMPTIED_IDX])
        self.assertEqual(1.0, features[MOVE_OCCUPIES_FREE_IDX])

    def test_move_features_released_free_cell_and_empty_column_target(self):
        game = FreeCellGame(deal=empty_deal())
        game.free_cells[0] = card(Suit.SPADES, 5)
        state = state_to_dict(game)

        features = move_to_features(
            state,
            {"move_type": "FREE_TO_COL", "from_idx": 0, "to_idx": 1, "count": 1},
        )

        self.assertEqual(1.0, features[MOVE_RELEASES_FREE_IDX])
        self.assertEqual(1.0, features[MOVE_TO_EMPTY_COLUMN_IDX])

    def test_move_features_release_low_card_and_increase_home(self):
        game = FreeCellGame(deal=empty_deal())
        game.columns[0] = [card(Suit.SPADES, 1), card(Suit.HEARTS, 9)]
        state = state_to_dict(game)

        free_features = move_to_features(
            state,
            {"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1},
        )
        home_game = FreeCellGame(deal=empty_deal())
        home_game.columns[0] = [card(Suit.SPADES, 1)]
        home_features = move_to_features(
            state_to_dict(home_game),
            {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
        )

        self.assertEqual(1.0, free_features[MOVE_RELEASES_LOW_IDX])
        self.assertEqual(1.0, home_features[MOVE_INCREASES_HOME_IDX])

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
