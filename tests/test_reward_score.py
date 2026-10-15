import unittest

from reward_score import (
    RewardScoreError,
    parse_reward_score,
    request_reward_score,
)


class ParseRewardScoreTests(unittest.TestCase):
    def test_parses_score_fence(self):
        self.assertEqual(0.75, parse_reward_score("```SCORE\n0.75\n```"))

    def test_parses_score_fence_surrounded_by_explanation(self):
        self.assertEqual(
            0.75,
            parse_reward_score("Evaluation complete.\n```SCORE\n0.75\n```\nDone."),
        )

    def test_parses_bare_number(self):
        self.assertEqual(0.5, parse_reward_score("0.5"))

    def test_parses_labelled_score(self):
        self.assertEqual(0.82, parse_reward_score("The score is 0.82."))

    def test_rejects_unrelated_number(self):
        with self.assertRaisesRegex(ValueError, "recognized score"):
            parse_reward_score("The experiment completed 12 runs.")

    def test_rejects_out_of_range_score(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            parse_reward_score("score: 1.2")

    def test_rejects_ambiguous_scores(self):
        with self.assertRaisesRegex(ValueError, "multiple different scores"):
            parse_reward_score("score: 0.4; revised score: 0.6")


class RequestRewardScoreTests(unittest.TestCase):
    def test_retries_only_the_reward_request(self):
        responses = iter(["not a score", "score: 0.6"])
        calls = []

        def request_score():
            calls.append(True)
            return next(responses)

        self.assertEqual(0.6, request_reward_score(request_score, attempts=3))
        self.assertEqual(2, len(calls))

    def test_raises_distinct_error_after_bounded_attempts(self):
        calls = []

        def request_score():
            calls.append(True)
            return "no numeric reward"

        with self.assertRaisesRegex(RewardScoreError, "after 3 attempts"):
            request_reward_score(request_score, attempts=3)
        self.assertEqual(3, len(calls))


if __name__ == "__main__":
    unittest.main()
