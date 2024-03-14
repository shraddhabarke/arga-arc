(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_color v G)
                (is_neighbor this v)
            )
        )
        (apply
            (extend_node v false)
        )
    )
)