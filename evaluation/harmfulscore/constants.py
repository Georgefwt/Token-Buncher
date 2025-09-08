"""Constant variables."""

__all__ = [
    'IGNORE_INDEX',
    'DEFAULT_BOS_TOKEN',
    'DEFAULT_EOS_TOKEN',
    'DEFAULT_PAD_TOKEN',
    'DEFAULT_UNK_TOKEN',
    'PROMPT_BEGIN',
    'PROMPT_USER',
    'PROMPT_ASSISTANT',
    'PROMPT_INPUT',
    'PROMPT_DICT',
    'ADAM_BETAS',
    'NUM_LABELS',
    'LABEL_NAMES',
    'ID2LABELS',
]


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)

LABEL_NAMES: list[str] = [
    'animal_abuse',
    'child_abuse',
    'controversial_topics,politics',
    'discrimination,stereotype,injustice',
    'drug_abuse,weapons,banned_substance',
    'financial_crime,property_crime,theft',
    'hate_speech,offensive_language',
    'misinformation_regarding_ethics,laws_and_safety',
    'non_violent_unethical_behavior',
    'privacy_violation',
    'self_harm',
    'sexually_explicit,adult_content',
    'terrorism,organized_crime',
    'violence,aiding_and_abetting,incitement',
]

NUM_LABELS: int = len(LABEL_NAMES)
ID2LABELS: dict[int, str] = dict(enumerate(LABEL_NAMES))