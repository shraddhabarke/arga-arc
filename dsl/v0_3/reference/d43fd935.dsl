(do
    (rule
        (filter
            (and
                (filter_by_neighbor_size max)
                (varand
                    (is_direct_neighbor)
                    (filter_by_color G)
                )
            )
        )
        (apply
            (extend_node var_extend_node true)
        )
    )
)