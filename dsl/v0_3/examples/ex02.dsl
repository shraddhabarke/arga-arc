(do
    (rule
        (filter
            (and
                (color_equals obj_color X)
                (size_equals obj_size min)
            )
        )
        (apply
            (update_color Y)
        )
    )
    (rule
        (filter
            (and
                (color_equals obj_color X)
                (size_equals obj_size min)
            )
        )
        (apply
            (update_color F)
        )
    )
    (rule
        (filter
            (and
                (color_equals obj_color X)
                (size_equals obj_size min)
            )
        )
        (apply
            (update_color A)
        )
    )
)