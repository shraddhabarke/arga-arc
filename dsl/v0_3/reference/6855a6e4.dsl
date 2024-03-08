(do
    (rule
        (with this v)
        (filter
            (and
                (is_color this X)
                (and
                    (is_color v R)
                    (is_neighbor this v)
                )
            )
        )
        (apply
            (mirror v)
        )
    )
)