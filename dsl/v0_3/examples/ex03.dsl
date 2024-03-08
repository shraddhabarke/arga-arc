(do
    (rule
        (with this)
        (filter)
        (apply
            (update_color R)
        )
    )
    (rule
        (with this)
        (filter
            (filter_by_size this 1)
        )
        (apply
            (update_color R)
        )
    )
)