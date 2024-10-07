import re
import json
from typing import Dict



def load_contractions(file_path: str) -> Dict[str, str]:
    '''
    Load contractions from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing contractions.

    Returns:
        Dict[str, str]: A dictionary mapping contractions to their expanded forms.
    '''
    with open(file_path, 'r') as f:
        return json.load(f)


def expand_contractions(text: str, contractions_dict: Dict[str, str]) -> str:
    '''
    Expand contractions in a given text using a provided contractions dictionary.

    Args:
        text (str): The input text containing contractions.
        contractions_dict (Dict[str, str]): A dictionary mapping contractions to their expanded forms.

    Returns:
        str: The text with contractions expanded.
    '''
    # Compile the regular expression pattern for matching contractions
    contractions_pattern = re.compile(
        '|'.join(re.escape(key) for key in contractions_dict.keys()),
        flags=re.IGNORECASE
    )

    # Function to replace each contraction with its expanded form
    def replace(match):
        return contractions_dict[match.group(0).lower()]

    # Substitute contractions in the text using the replace function
    return contractions_pattern.sub(replace, text)
