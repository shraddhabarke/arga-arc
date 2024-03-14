(do
    (rule
        (with this v)
        (filter
            (and
                (or
                    (filter_by_color v R)
                    (filter_by_color v B)
                )
                (is_neighbor this v)
            )
        )
        (apply
            (move_node_max v)
        )
    )
)