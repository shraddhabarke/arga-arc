(do
    (rule
        (with this v)
        (filter
            (and
                (is_color v G)
                (is_neighbor this v)
            )
        )
        (apply
            (extend_node v false)
        )
    )
)