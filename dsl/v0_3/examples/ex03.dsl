(do
    (rule
        (filter)
        (apply
            (update_color R)
        )
    )
    (rule
        (filter
            (size_equals obj_size 1)
        )
        (apply
            (update_color R)
        )
    )
)