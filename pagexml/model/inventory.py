import re


def parse_number(number: str):
    """Parse an inventory number by identifying parts that use a different sorting
    (e.g. alphabetic versus numeric)."""
    if isinstance(number, str) is False:
        return None
    if len(number) == 0:
        return []
    parts = []
    if number[0].isdigit():
        match = re.match(r"(\d+)", number)
        part = match.group(1)
        parts.append(part)
        number = number[len(part):]
    elif number[0].isalpha():
        match = re.match(r"([A-Za-z])", number)
        part = match.group(1)
        parts.append(part)
        number = number[len(part):]
    elif number[0] == '.':
        part = number[0]
        parts.append(part)
        number = number[len(part):]
    if len(number) > 0:
        other_parts = parse_number(number)
        parts.extend(other_parts)
    return parts
