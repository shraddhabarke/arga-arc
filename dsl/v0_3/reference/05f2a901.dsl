(do
    (rule
        (with this v)
        (filter
            (and
                (is_color this R)
                (is_color v C)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)