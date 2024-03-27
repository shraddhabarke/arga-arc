(do
    (rule
        (filter
            (and
                (filter_by_size 1)
                (varand
                    (is_direct_neighbor)
                    (filter_by_color X)
                )
            )
        )
        (apply
            (update_color R)
            (move_node_max down)
        )
    )
    (rule
        (filter
            (and
                (filter_by_size 1)
                (varand
                    (is_direct_neighbor)
                    (filter_by_color Y)
                )
            )
        )
        (apply
            (update_color B)
            (move_node_max up)
        )
    )
)