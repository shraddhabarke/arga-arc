(do
    (rule
        (vars (this var))
        (filter
            (and
                (neighbor_of this var)
                (and
                    (color_equals (color_of var) G)
                    (size_equals (size_of var) max)
                )
            )
        )
        (apply
            (extend_node (direction_of var) true)
        )
    )
)