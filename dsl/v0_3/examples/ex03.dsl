(do
    (rule
        (filter)
        (apply
            (update_color R)
        )
    )
    (rule
        (filter
            (filter_by_size 1)
        )
        (apply
            (update_color R)
        )
    )
)