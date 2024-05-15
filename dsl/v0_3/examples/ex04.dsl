(do
    (rule
        (vars (this other))
        (filter
            (and
                (neighbor_of this other)
                (size_equals (size_of other) 4)
            )
        )
        (apply
            (move_node_max (direction_of other))
        )
    )
)