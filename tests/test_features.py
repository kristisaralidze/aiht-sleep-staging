import numpy as np
import pytest

from src import features

COUNTS = features.CountParams(fs_hz=50, band_hz=(3.0, 11.0), filter_order=5, n_bins=128,
                              bin_range=(0.0, 5.0), count_seconds=15, offset=18.0, scale=3.07)
HEART = features.HeartRateParams(dog_sigmas_s=(120.0, 600.0), dog_scalar=0.75, dog_points=285,
                                 norm_percentile=90.0)
WINDOW = features.WindowParams(half_width_s=285, count_sigma_s=50.0)
CLOCK = features.ClockParams(shift_hours=5.0)
PARAMS = features.FeatureParams(epoch_s=30, counts=COUNTS, heart=HEART, window=WINDOW, clock=CLOCK)


def _sine(frequency, amplitude, seconds=120, fs=50):
    times = np.arange(0, seconds, 1 / fs)
    return times, amplitude * np.sin(2 * np.pi * frequency * times)


def test_counts_are_zero_for_a_still_watch():
    times = np.arange(0, 120, 0.02)
    _, counts = features.activity_counts(times, np.full(times.shape, -1.0), 0, 120, COUNTS)
    np.testing.assert_array_equal(counts, np.zeros(8))


def test_counts_ignore_movement_below_the_band():
    times, z = _sine(0.2, 1.0)
    _, counts = features.activity_counts(times, z, 0, 120, COUNTS)
    assert counts[1:-1].max() == 0


def test_counts_match_hand_calculation_for_in_band_sine():
    times, z = _sine(6.0, 1.0)
    _, counts = features.activity_counts(times, z, 0, 120, COUNTS)
    peak_bin = np.floor(1.0 / 5.0 * 128) + 1
    expected = (15 * peak_bin - 18) * 3.07
    np.testing.assert_allclose(counts[2:-2], expected, rtol=0.03)


def test_counts_grow_with_movement_amplitude():
    times, small = _sine(6.0, 0.2)
    _, large = _sine(6.0, 1.0)
    low = features.activity_counts(times, small, 0, 120, COUNTS)[1]
    high = features.activity_counts(times, large, 0, 120, COUNTS)[1]
    assert high[3] > low[3] > 0


def test_count_times_are_bin_centres():
    times = np.arange(0, 60, 0.02)
    count_times, _ = features.activity_counts(times, np.zeros(times.shape), 0, 60, COUNTS)
    np.testing.assert_array_equal(count_times, [7.5, 22.5, 37.5, 52.5])


def test_resample_covers_the_interval_at_one_hertz():
    grid, values = features.resample_1hz(np.array([0.0, 10.0]), np.array([0.0, 10.0]), 0, 11)
    np.testing.assert_array_equal(grid, np.arange(11.0))
    np.testing.assert_array_equal(values, np.array([*range(11)], dtype=float))


def test_gaussian_mean_of_constant_is_the_constant():
    assert features.gaussian_mean(np.full(599, 3.0), 50.0) == pytest.approx(3.0)


def test_gaussian_mean_weights_the_centre_most():
    centre = np.zeros(599)
    centre[299] = 1.0
    edge = np.zeros(599)
    edge[0] = 1.0
    assert features.gaussian_mean(centre, 50.0) > features.gaussian_mean(edge, 50.0)


def test_dog_filter_keeps_length_and_removes_constant():
    filtered = features.dog_filter(np.full(1000, 70.0), HEART)
    assert filtered.shape == (1000,)
    np.testing.assert_allclose(filtered, 0.0, atol=1e-9)


def test_normalised_hr_is_invariant_to_subject_baseline_and_scale():
    rng = np.random.default_rng(0)
    heart_rate = rng.normal(size=3000).cumsum()
    base = features.normalise_filtered_hr(heart_rate, HEART)
    shifted = features.normalise_filtered_hr(3.0 * heart_rate + 40.0, HEART)
    np.testing.assert_allclose(base, shifted, atol=1e-9)


def test_window_slice_is_strict_and_spans_both_sides_of_the_epoch():
    selected = features.window_slice(np.arange(0.0, 1000.0), 300.0, 30.0, 285)
    assert (selected.start, selected.stop) == (16, 615)


def test_cosine_proxy_troughs_at_shift_hours():
    assert features.cosine_proxy(np.array([5 * 3600.0]), CLOCK)[0] == pytest.approx(-1.0)
    assert features.cosine_proxy(np.array([17 * 3600.0]), CLOCK)[0] == pytest.approx(1.0)


def _night(seconds=1800):
    rng = np.random.default_rng(1)
    motion_times = np.arange(0, seconds, 0.02)
    z = rng.normal(scale=0.05, size=motion_times.shape) - 1.0
    hr_times = np.arange(0, seconds, 5.0)
    heart_rate = 60 + 5 * np.sin(hr_times / 200.0) + rng.normal(scale=0.5, size=hr_times.shape)
    return motion_times, z, hr_times, heart_rate


def _build(epoch_times, z_change=None, seconds=1800):
    motion_times, z, hr_times, heart_rate = _night(seconds)
    if z_change is not None:
        z = z + z_change(motion_times)
    return features.build_features(epoch_times, motion_times, z, hr_times, heart_rate,
                                   start=0.0, end=float(seconds), params=PARAMS)


def test_build_features_shape_and_names():
    epoch_times = np.arange(0.0, 1800.0, 30.0)
    matrix, names, groups = _build(epoch_times)
    assert matrix.shape == (60, len(names))
    assert len(names) == len(groups)
    assert np.isfinite(matrix).all()
    assert set(groups) == {"motion", "heart_rate", "time"}


def test_time_features_start_at_first_valid_epoch():
    matrix, names, _ = _build(np.array([60.0, 90.0, 1770.0]))
    column = matrix[:, names.index("time_hours")]
    assert column[0] == 0.0
    assert column[1] == pytest.approx(30 / 3600)


def _burst(times):
    """Strong in-band movement between 1000 s and 1100 s."""
    return np.where((times >= 1000) & (times < 1100), np.sin(2 * np.pi * 6 * times), 0.0)


def test_movement_outside_an_epoch_reaches_windowed_but_not_epoch_counts():
    epoch_times = np.array([900.0])
    before, names, _ = _build(epoch_times)
    after, _, _ = _build(epoch_times, z_change=_burst)
    assert after[0, names.index("count_walch")] > before[0, names.index("count_walch")]
    assert after[0, names.index("count_epoch")] == before[0, names.index("count_epoch")]


def test_movement_inside_an_epoch_raises_its_epoch_count():
    epoch_times = np.array([1020.0])
    before, names, _ = _build(epoch_times)
    after, _, _ = _build(epoch_times, z_change=_burst)
    assert after[0, names.index("count_epoch")] > before[0, names.index("count_epoch")]
