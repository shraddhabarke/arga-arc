(do
    (rule
        (with this v)
        (filter
            (and
                (not (filter_by_size this 1))
                (filter_by_size v 1)
            )
        )
        (apply
            (insert v TOP TARGET)
        )
    )
)