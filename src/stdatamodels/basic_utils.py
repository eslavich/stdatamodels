"""General utility objects"""
import re


def bytes2human(n):
    """Convert bytes to human-readable format

    Taken from the `psutil` library which references
    http://code.activestate.com/recipes/578019

    Parameters
    ----------
    n : int
        Number to convert

    Returns
    -------
    readable : str
        A string with units attached.

    Examples
    --------
    >>> bytes2human(10000)
        '9.8K'

    >>> bytes2human(100001221)
        '95.4M'
    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n


def multiple_replace(string, rep_dict):
    """Single-pass replacement of multiple substrings

    Similar to `str.replace`, except that a dictionary of replacements
    can be specified.

    The replacements are done in a single-pass. This means that a previous
    replacement will not be replaced by a subsequent match.

    Parameters
    ----------
    string: str
        The source string to have replacements done on it.

    rep_dict: dict
        The replacements were key is the input substring and
        value is the replacement

    Returns
    -------
    replaced: str
        New string with the replacements done

    Examples
    --------
    Basic example that also demonstrates the single-pass nature.
    If the replacements where chained, the result would have been
    'lamb lamb'

    >>> multiple_replace('button mutton', {'but': 'mut', 'mutton': 'lamb'})
    'mutton lamb'

    """
    pattern = re.compile(
        "|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]),
        flags=re.DOTALL
    )
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)
