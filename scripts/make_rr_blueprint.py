import rerun.blueprint as rrb


def make_blueprint():
    my_blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(
                            origin="/cameras",
                            visual_bounds=rrb.VisualBounds2D(
                                # Hard-coded camera count
                                x_range=[0, 2],
                                y_range=[0, 1],
                            ),
                        ),
                        rrb.Spatial2DView(
                            origin="/world",
                            visual_bounds=rrb.VisualBounds2D(
                                # Hard-coded world area
                                x_range=[-114, 11],
                                y_range=[-64, 40],
                            ),
                        ),
                    ]
                ),
                rrb.TimeSeriesView(
                    name="Processing times (msec)",
                    origin="/processing_times",
                    axis_y=rrb.ScalarAxis(range=(0, 200), zoom_lock=True),
                ),
            ],
            row_shares=[0.8, 0.2],
        ),
        collapse_panels=False,
    )
    return my_blueprint


if __name__ == "__main__":
    bp = make_blueprint()
    bp.save("zoo_vision", "data/zoo_vision.rbl")
