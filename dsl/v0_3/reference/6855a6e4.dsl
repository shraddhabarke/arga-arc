(do
    (rule
        (vars this var)
        (filter
            (and
                (color_equals (color_of this) X)     
                (and
                    (neighbor_of this var)
                    (color_equals (color_of var) R)
                )
            )
        )
        (apply
            (mirror var)
        )
    )
)