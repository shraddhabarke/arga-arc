(do
    (rule
        (vars (this x))
        (filter
            (and
                (size_equals (size_of this) 1)
                (and
                    (neighbor_of this x)
                    (size_equals (size_of x) max)
                )
            )
        )
        (apply
            (extend_node (direction_of x) false)
        )    
    )
)