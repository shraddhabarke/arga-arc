(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_color this X)
                (and
                    (filter_by_color v R)
                    (is_neighbor this v)
                )
            )
        )
        (apply
            (mirror v)
        )
    )
)