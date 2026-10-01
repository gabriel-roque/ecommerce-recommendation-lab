from app.training.split import temporal_split


def test_temporal_split_keeps_all_events_and_prevents_timestamp_leakage():
    timestamps = list(range(1, 9)) + [9] * 4 + [10, 11, 12, 13]
    events = [{"timestamp": timestamp, "row": row} for row, timestamp in enumerate(timestamps)]

    split = temporal_split(events)

    assert len(split.train) + len(split.validation) + len(split.test) == len(events)
    assert {event["row"] for event in split.train + split.validation + split.test} == set(
        range(len(events))
    )
    assert max(event["timestamp"] for event in split.train) < min(
        event["timestamp"] for event in split.validation
    )
    assert max(event["timestamp"] for event in split.validation) < min(
        event["timestamp"] for event in split.test
    )


def test_temporal_split_requires_three_distinct_timestamps():
    events = [{"timestamp": value} for value in [1, 1, 2, 2]]
    try:
        temporal_split(events)
    except ValueError as error:
        assert "distinct timestamps" in str(error)
    else:
        raise AssertionError("Leaky split was accepted")
