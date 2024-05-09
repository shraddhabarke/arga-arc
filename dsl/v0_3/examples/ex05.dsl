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
            (extend_node (direction_of x) false)
        )
    )
)