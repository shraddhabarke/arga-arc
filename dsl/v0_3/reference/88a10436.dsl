(do
    (rule
        (with this v)
        (filter
            (and
                (not (is_size this 1))
                (is_size v 1)
            )
        )
        (apply
            (insert v)
        )
    )
)