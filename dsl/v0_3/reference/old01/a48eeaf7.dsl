(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_color this X)
                (filter_by_color v R)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)