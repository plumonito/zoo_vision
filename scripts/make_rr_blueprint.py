import rerun.blueprint as rrb


def make_blueprint():
    my_blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(origin="/cameras"),
                        rrb.Spatial2DView(origin="/world"),
                    ]
                ),
                rrb.TimeSeriesView(
                    name="Processing times (msec)",
                    origin="/processing_times",
                    # Set a custom Y axis.
                    axis_y=rrb.ScalarAxis(range=(0, 100), zoom_lock=True),
                    # Configure the legend.
                    # plot_legend=rrb.PlotLegend(visible=False),
                    # Set time different time ranges for different timelines.
                    # time_ranges=[
                    # Sliding window depending on the time cursor for the first timeline.
                    # rrb.VisibleTimeRange(
                    #     "ros_time",
                    #     start=rrb.TimeRangeBoundary.cursor_relative(seq=-200),
                    #     end=rrb.TimeRangeBoundary.cursor_relative(),
                    # ),
                    # # Time range from some point to the end of the timeline for the second timeline.
                    # rrb.VisibleTimeRange(
                    #     "timeline1",
                    #     start=rrb.TimeRangeBoundary.absolute(seconds=300.0),
                    #     end=rrb.TimeRangeBoundary.infinite(),
                    # ),
                    # ],
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
