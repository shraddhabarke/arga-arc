(do
    (rule
        (vars this var)
        (filter
                (and
                    (neighbor_of this var)
                    (color_equals (color_of var) G)
                    (size_equals (size_of var) max)
                )
            )
        )
        (apply
            (extend_node vardirection true)
        )
    )
)