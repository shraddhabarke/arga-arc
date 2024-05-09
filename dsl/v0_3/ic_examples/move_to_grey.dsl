(do
    (rule
        (vars (this x))
        (filter
            (and
                (not (color_equals (color_of this) X))
                (and
                    (neighbor_of this x)
                    (color_equals (color_of x) X)
                )
            )
        )
        (apply
            (move_node_max (direction_of x))
        )
    )
)