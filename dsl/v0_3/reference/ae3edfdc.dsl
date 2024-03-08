(do
    (rule
        (with this v)
        (filter
            (and
                (or
                    (is_color v R)
                    (is_color v B)
                )
                (is_neighbor this v)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)