(do
    (rule
        (with this x)
        (filter
            (and
                (filter_by_size this 1)
                (filter_by_color x X)
                (is_neighbor this x)
            )
        )
        (apply
            (extend_node x false)
        )
    )
)