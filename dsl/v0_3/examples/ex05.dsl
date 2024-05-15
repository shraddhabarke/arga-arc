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
            (extend_node (direction_of other) false)
        )
    )
)