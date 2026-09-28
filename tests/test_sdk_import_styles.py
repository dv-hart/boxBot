"""Every documented way of reaching the SDK must yield the same module.

Regression for the intermittent in-the-field fumble: the model writes one of
three import styles per script, and ``from boxbot_sdk import bb`` used
to ImportError (the alias lived only in sys.modules, not as a package
attribute) — the first lock command of a conversation then failed.

In the sandbox venv the package is copied as top-level ``boxbot_sdk``;
on the dev box it is ``boxbot.sdk`` — the aliasing lives in the shared
``__init__`` either way, so testing through ``boxbot.sdk`` covers both.
"""

import sys


def test_bb_attribute_and_alias_all_point_at_the_sdk():
    import boxbot.sdk as sdk
    from boxbot.sdk import bb as bb_attr  # the style that used to crash

    assert bb_attr is sdk
    assert sdk.bb is sdk
    # ``import bb`` inside a sandbox script resolves via this alias.
    assert sys.modules["bb"] is sdk
    # One transport singleton regardless of import style.
    assert bb_attr.workspace is sdk.workspace
