(do
    (rule
        (vars (this x))
        (filter
            (and
                (size_equals (size_of this) 1)
                (and
                    (neighbor_of this x)
                    (color_equals (color_of x) X)
                )
            )
        )
        (apply
            (update_color R)
            (move_node_max down)
        )
    )
    (rule
        (vars (this x))
        (filter
            (and
                (size_equals (size_of this) 1)
                (and
                    (neighbor_of this x)
                    (color_equals (color_of x) Y)
                )
            )
        )
        (apply
            (update_color B)
            (move_node_max up)
        )
    )
)