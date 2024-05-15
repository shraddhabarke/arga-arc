(do
    (rule
        (vars (this other))
        (filter
            (and
                (size_equals (size_of this) 1)
                (and
                    (neighbor_of this other)
                    (color_equals (color_of other) X)
                )
            )
        )
        (apply
            (update_color R)
            (move_node_max down)
        )
    )
    (rule
        (vars (this other))
        (filter
            (and
                (size_equals (size_of this) 1)
                (and
                    (neighbor_of this other)
                    (color_equals (color_of other) Y)
                )
            )
        )
        (apply
            (update_color B)
            (move_node_max up)
        )
    )
)