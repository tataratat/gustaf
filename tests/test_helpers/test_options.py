import pytest

from gustaf.helpers import options


def backend():
    return "vedo"


def key():
    return "color"


def description():
    return "color in german is Farbe"


def types():
    return (str, tuple, list)


def default():
    return "green"


def option():
    return options.Option(backend(), key(), description(), types(), default())


def test_option_init():
    op = option()
    assert backend() in op.backends
    assert op.key == key()
    assert op.description == description()
    assert op.allowed_types == types()
    assert op.default == default()


def test_option_type_check():
    wrong_type_default = 1
    assert type(wrong_type_default) not in types()

    with pytest.raises(TypeError):
        options.Option(1, key(), description(), types(), default())
    with pytest.raises(TypeError):
        options.Option(backend(), 1, description(), types(), default())
    with pytest.raises(TypeError):
        options.Option(backend(), key(), 1, types(), default())
    with pytest.raises(TypeError):
        options.Option(backend(), key(), description(), 1, default())
    with pytest.raises(TypeError):
        options.Option(
            backend(), key(), description(), types(), wrong_type_default
        )


def test_make_valid_options():
    option0 = option()
    option1 = options.Option(
        "backend", "key", "description", (int, float), 100
    )
    valid_opt = options.make_valid_options(
        option0,
        option1,
    )

    # vo should make a deepcopy
    assert id(valid_opt[option0.key]) != id(option0)
    assert id(valid_opt[option1.key]) != id(option1)

    o1_default = -100
    valid_opt_default_overwrite = options.make_valid_options(
        option0, option1, options.SetDefault(option1.key, o1_default)
    )

    assert valid_opt_default_overwrite[option1.key].default == o1_default
    # o0 default stays the same
    assert valid_opt_default_overwrite[option0.key].default == default()

    with pytest.raises(TypeError):
        valid_opt_default_overwrite = options.make_valid_options(
            option0,
            option1,
            options.SetDefault(option1.key, "str is not allowed"),
        )
