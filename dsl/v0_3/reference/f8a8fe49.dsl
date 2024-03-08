(do
    (rule
        (with this v)
        (filter
            (and
                (is_color this X)
                (is_neighbor this v)
            )
        )
        (apply
            (mirror v)
        )
    )
)