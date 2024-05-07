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
            (update_color R)
            (move_node_max down)
        )
    )
    (rule
        (filter
            (and
                (size_equals obj_size 1)
                (and
                    (neighbor_of obj)
                    (color_equals obj_color Y)
                )
            )
        )
        (apply
            (update_color B)
            (move_node_max up)
        )
    )
)