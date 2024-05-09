(do
    (rule
        (vars (this var))
        (filter
            (and
                (color_equals (color_of this) X)
                (and
                    (neighbor_of obj var)     
                    (color_equals (color_of var) R)
                )
            )
        )
        (apply
            (move_node_max (direction_of var))
        )
    )
)