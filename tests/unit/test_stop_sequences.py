from app.runtime.stop_sequences import StopSequenceFilter


def test_without_stops_text_passes_through() -> None:
    stops = StopSequenceFilter(None)
    assert stops.push("hello") == "hello"
    assert stops.finish() == ""


def test_stops_before_the_stop_text_split_across_pieces() -> None:
    stops = StopSequenceFilter(["END"])
    out = stops.push("one two E") + stops.push("N") + stops.push("D three")
    assert out == "one two "
    assert stops.stopped
    assert stops.push("more") == ""


def test_a_false_start_is_released() -> None:
    stops = StopSequenceFilter(["END"])
    assert stops.push("an E") == "an "
    assert stops.push("dge") == "Edge"
    assert not stops.stopped


def test_held_text_comes_out_at_the_end() -> None:
    stops = StopSequenceFilter(["###"])
    assert stops.push("total ##") == "total "
    assert stops.finish() == "##"


def test_earliest_of_several_stops_wins() -> None:
    stops = StopSequenceFilter(["b", "a"])
    assert stops.push("xxab") == "xx"
