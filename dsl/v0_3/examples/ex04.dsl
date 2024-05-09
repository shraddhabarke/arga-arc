(do
    (rule
        (vars (this x))
        (filter
            (and
                (neighbor_of this x)
                (size_equals (size_of x) 4)
            )
        )
        (apply
            (move_node_max (direction_of x))
        )
    )
)