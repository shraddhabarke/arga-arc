(do
    (rule
        (vars (this other))
        (filter
            (and
                (size_equals (size_of this) 1)
                (and
                    (neighbor_of this other)
                    (size_equals (size_of other) max)
                )
            )
        )
        (apply
            (extend_node (direction_of other) false)
        )    
    )
)