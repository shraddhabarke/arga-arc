(do
    (rule
        (with this v)
        (filter
            (and
                (is_color v G)
                (is_neighbor this v)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)