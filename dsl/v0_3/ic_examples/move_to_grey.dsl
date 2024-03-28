(do
    (rule
        (filter
            (and
                (not (filter_by_color X))
                (varand
                    (is_direct_neighbor)
                    (filter_by_color X)
                )
            )
        )
        (apply
            (move_node_max var_move_node_max)
        )
    )
)