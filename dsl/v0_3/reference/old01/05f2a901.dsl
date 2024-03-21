(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_color this R)
                (filter_by_color v C)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)