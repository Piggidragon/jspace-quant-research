from jspace_quant import config, data


def test_readout_position_poetry_is_last_newline():
    pieces = ["<s>", "A", " rhyming", ":\n", "The", " line", ",\n", "And", " face", " of", " "]
    assert data.readout_position("poetry", pieces) == 6
    assert data.readout_position("multihop", pieces) == len(pieces) - 1


def test_model_prompt_strips_only_trailing_spaces():
    assert data.model_prompt("is the ") == "is the"
    assert data.model_prompt("(2 + 3) * 4 = ") == "(2 + 3) * 4 ="
    assert data.model_prompt('es "') == 'es "'
    assert data.model_prompt("line,\nAnd face of ") == "line,\nAnd face of"


def test_surface_forms_cover_space_and_case_variants():
    forms = data.surface_forms("Brazil", "multihop")
    assert {"Brazil", " Brazil", "brazil", " brazil"} <= set(forms)
    assert len(forms) == len(set(forms))


def test_order_ops_expansion():
    assert {"5", " five"} <= set(data.surface_forms("5", "order-ops"))
    assert {"*", " times", "multiplication"} <= set(data.surface_forms("multiplication", "order-ops"))
    assert data.number_word(24) == "twenty-four"


def test_candidate_tokens_prefers_single_token_forms():
    vocab = {" Brazil": [7], "Brazil": [3, 4], " brazil": [9, 9]}

    def tok(s):
        return vocab.get(s, [1, 2])

    c = data.candidate_tokens("Brazil", "multihop", tok)
    assert c.single_token and c.token_ids == [7]


def test_candidate_tokens_fallback_when_no_single_token_form():
    c = data.candidate_tokens("Xyz", "typo", lambda s: [11, 12])
    assert not c.single_token and c.token_ids == [11]


def test_all_eval_sets_load_with_intermediates():
    for name in config.EVAL_SETS:
        items = data.load_eval_set(name)
        assert items and all(it.intermediates for it in items)


def test_order_ops_intermediates_all_expandable():
    for it in data.load_eval_set("order-ops"):
        for w in it.intermediates:
            assert data.surface_forms(w, "order-ops")


def test_heldout_positions_skip_sinks_and_last():
    pos = data.heldout_positions(128)
    assert pos[0] == config.HELDOUT_SKIP_FIRST and pos[-1] < 127
