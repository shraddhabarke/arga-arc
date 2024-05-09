(do
    (rule
        (filter
            (and
                (size_equals obj_size 1)
                (and
                    (neighbor_of obj)
                    (color_equals obj_color X)
                )
            )
        )
        (apply
            (extend_node vardirection false)
        )
    )
)