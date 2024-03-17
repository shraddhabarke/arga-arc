(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_color this X)
                (is_neighbor this v)
            )
        )
        (apply
            (mirror v)
        )
    )
)