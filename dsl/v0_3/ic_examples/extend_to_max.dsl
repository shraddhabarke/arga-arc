(do
    (rule
        (filter
            (and
                (filter_by_size 1)
                (varand
                    (is_direct_neighbor)
                    (filter_by_size max)
                )
            )
        )
        (apply
            (extend_node var_extend_node false)
        )    
    )
)