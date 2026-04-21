from app.planning.dstar_lite import DStarLite


def test_dstar_lite_plans_path_around_obstacles():
    planner = DStarLite(
        start=(0, 0),
        goal=(4, 4),
        obs={(1, 0), (1, 1), (1, 2), (2, 2), (3, 2)},
        x_max=6,
        y_max=6,
        inflation_radius=0,
        penalty_weight=0.0,
    )

    path = planner.plan()

    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert not any(node in planner.obs for node in path)


def test_dstar_lite_avoids_passable_terrain_when_alternative_exists():
    planner = DStarLite(
        start=(0, 1),
        goal=(4, 1),
        obs=set(),
        x_max=5,
        y_max=3,
        inflation_radius=0,
        penalty_weight=0.0,
        terrain_penalties={(1, 1): 10.0, (2, 1): 10.0, (3, 1): 10.0},
    )

    path = planner.plan()

    assert path[0] == (0, 1)
    assert path[-1] == (4, 1)
    assert not {(1, 1), (2, 1), (3, 1)}.issubset(set(path))



def test_dstar_lite_can_cross_passable_terrain_when_needed():
    planner = DStarLite(
        start=(0, 1),
        goal=(2, 1),
        obs={(1, 0), (1, 2)},
        x_max=3,
        y_max=3,
        inflation_radius=0,
        penalty_weight=0.0,
        terrain_penalties={(1, 1): 10.0},
    )

    path = planner.plan()

    assert path[0] == (0, 1)
    assert path[-1] == (2, 1)
    assert (1, 1) in path


