from src import experiment


def test_model_grid_merges_fixed_params_into_each_candidate():
    grid = experiment.model_grid({"fixed": {"n": 100}, "grid": {"a": [1, 2]}})
    assert grid == [{"n": 100, "a": 1}, {"n": 100, "a": 2}]


def test_model_grid_accepts_a_list_of_sub_grids_without_crossing_them():
    config = {"fixed": {}, "grid": [{"a": [1, 2], "b": [3]}, {"a": [None], "b": [4]}]}
    grid = experiment.model_grid(config)
    assert len(grid) == 3
    assert {"a": None, "b": 4} in grid
    assert {"a": None, "b": 3} not in grid
