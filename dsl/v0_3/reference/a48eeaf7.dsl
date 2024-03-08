(do
    (rule
        (with this v)
        (filter
            (and
                (is_color this X)
                (is_color v R)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)