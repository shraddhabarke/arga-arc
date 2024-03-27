(do
    (rule
        (filter
            (and
                (filter_by_size 1)
                (varand
                    (is_any_neighbor)
                    (filter_by_color X)
                )
            )
        )
        (apply
            (extend_node var_extend_node false)
        )
    )
)