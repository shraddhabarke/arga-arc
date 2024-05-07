(do
    (rule
        (filter
            (and
                (neighbor_size obj_size max)
                (and
                    (neighbor_of obj)
                    (color_equals obj_color G)
                )
            )
        )
        (apply
            (extend_node vardirection true)
        )
    )
)