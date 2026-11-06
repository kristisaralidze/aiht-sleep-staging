import numpy as np
import pytest

from src import epochs


def test_map_stages_three_classes_including_n4():
    stages = np.array([0, 1, 2, 3, 4, 5, -1])
    expected = [epochs.WAKE, epochs.NREM, epochs.NREM, epochs.NREM, epochs.NREM, epochs.REM,
                epochs.UNSCORED]
    np.testing.assert_array_equal(epochs.map_stages(stages), expected)


def test_class_codes_match_spec():
    assert (epochs.WAKE, epochs.NREM, epochs.REM) == (0, 1, 2)


def test_map_stages_rejects_unknown_stage():
    with pytest.raises(ValueError, match="7"):
        epochs.map_stages(np.array([0, 7]))


def test_crop_interval_is_intersection_of_all_recordings():
    labels = np.array([0.0, 30.0, 60.0])
    motion = np.array([-100.0, 70.0])
    heart_rate = np.array([10.0, 500.0])
    assert epochs.crop_interval(labels, motion, heart_rate, 30) == (10.0, 70.0)


def test_crop_interval_includes_full_final_epoch():
    labels = np.array([0.0, 30.0])
    wide = np.array([-100.0, 1000.0])
    assert epochs.crop_interval(labels, wide, wide, 30) == (0.0, 60.0)


def test_crop_interval_raises_when_recordings_do_not_overlap():
    with pytest.raises(ValueError):
        epochs.crop_interval(np.array([0.0, 30.0]), np.array([100.0, 200.0]),
                             np.array([0.0, 50.0]), 30)


def test_samples_per_epoch_uses_half_open_bins():
    starts = np.array([0.0, 30.0])
    samples = np.array([0.0, 29.99, 30.0, 59.0, 60.0])
    np.testing.assert_array_equal(epochs.samples_per_epoch(samples, starts, 30), [2, 2])


def _select(stages, motion, heart_rate):
    times = np.arange(len(stages)) * 30.0
    return epochs.select_epochs(times, np.array(stages), np.array(motion), np.array(heart_rate), 30)


def test_select_epochs_drops_unscored():
    signal = np.arange(0.0, 90.0, 5.0)
    selection = _select([0, -1, 5], signal, signal)
    np.testing.assert_array_equal(selection.times, [0.0, 60.0])
    np.testing.assert_array_equal(selection.labels, [epochs.WAKE, epochs.REM])
    assert selection.n_unscored == 1
    assert selection.n_no_signal == 0


def test_select_epochs_drops_epoch_missing_either_signal():
    full = np.arange(0.0, 120.0, 5.0)
    motion_gap = full[(full < 30) | (full >= 60)]
    hr_gap = full[(full < 60) | (full >= 90)]
    selection = _select([1, 1, 1, 1], motion_gap, hr_gap)
    np.testing.assert_array_equal(selection.times, [0.0, 90.0])
    assert selection.n_no_signal == 2


def test_unscored_epoch_without_signal_is_counted_once_as_unscored():
    signal = np.array([0.0, 5.0])
    selection = _select([0, -1], signal, signal)
    assert (selection.n_unscored, selection.n_no_signal) == (1, 0)


def test_count_sequence_breaks_finds_non_adjacent_epochs():
    times = np.array([0.0, 30.0, 120.0, 150.0, 300.0])
    assert epochs.count_sequence_breaks(times, 30) == 2
