(do
    (rule
        (vars (this other))
        (filter
            (and
                (neighbor_of this other)
                (and
                    (color_equals (color_of other) G)
                    (size_equals (size_of other) max)
                )
            )
        )
        (apply
            (extend_node (direction_of other) true)
        )
    )
)