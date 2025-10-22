import pytest

from training.simplification import simplify


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Schneiderin", "schneider"),
        ("Schneider/in", "schneider"),
        ("Mitarbeiter*innen", "mitarbeiter"),
        ("Kund:innen", "kund"),
        ("LehrerInnen", "lehrer"),
    ],
)
def test_simplify_gender_normalization_examples(raw, expected):
    """ Tests simplify() with gender normalization examples """
    out = simplify(raw, do_stemming=False)  # Without stemming, to test only gender normalization
    assert out == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Mädchen", "maedch"),  # umlaut replacement + lowercase
        ("Straße", "strass"),  # ß -> ss
        ("  ÜBERGABE  ", "uebergab"),  # strip + lowercase + umlaut map
    ],
)
def test_simplify(raw, expected):
    """
    Tests simplify().

    Expected: word stem
    """
    out = simplify(raw, do_stemming=True)
    assert out == expected
