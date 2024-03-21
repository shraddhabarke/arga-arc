(do
    (rule
        (filter
            (and
                (filter_by_color X)
                (varand
                    (is_any_neighbor)     
                    (filter_by_color R)
                )
            )
        )
        (apply
            (move_node_max var)
        )
    )
)