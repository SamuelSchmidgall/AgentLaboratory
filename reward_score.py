import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


class RewardScoreError(RuntimeError):
    """Raised when the reward model does not produce a usable score."""


def _validate_score(value):
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"reward score must be between 0 and 1, got {score}")
    return score


def parse_reward_score(response):
    """Parse a reward score without accepting unrelated numbers in prose."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("reward response is empty")

    text = response.strip()
    fenced = re.findall(
        rf"```SCORE\s*({_NUMBER})\s*```", text, flags=re.IGNORECASE
    )
    unique_fenced_scores = list(dict.fromkeys(fenced))
    if len(unique_fenced_scores) == 1:
        return _validate_score(unique_fenced_scores[0])
    if len(unique_fenced_scores) > 1:
        raise ValueError("reward response contains multiple different scores")

    unlabelled_fence = re.fullmatch(rf"```\s*({_NUMBER})\s*```", text)
    if unlabelled_fence:
        return _validate_score(unlabelled_fence.group(1))

    bare = re.fullmatch(rf"({_NUMBER})", text)
    if bare:
        return _validate_score(bare.group(1))

    labelled = re.findall(
        rf"\bscore\b\s*(?:(?:is)\s*)?(?:[:=]\s*)?({_NUMBER})(?=$|[\s,;.!?])",
        text,
        flags=re.IGNORECASE,
    )
    unique_scores = list(dict.fromkeys(labelled))
    if len(unique_scores) == 1:
        return _validate_score(unique_scores[0])
    if len(unique_scores) > 1:
        raise ValueError("reward response contains multiple different scores")
    raise ValueError("reward response does not contain a recognized score")


def request_reward_score(request_score, attempts=3):
    """Retry only reward generation/parsing and return a validated score."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    errors = []
    last_response = None
    for _ in range(attempts):
        try:
            last_response = request_score()
            return parse_reward_score(last_response)
        except Exception as exc:
            errors.append(str(exc))

    response_preview = repr(last_response)
    if len(response_preview) > 500:
        response_preview = response_preview[:497] + "..."
    raise RewardScoreError(
        f"Reward scoring failed after {attempts} attempts. "
        f"Last error: {errors[-1]}. Last response: {response_preview}"
    )
