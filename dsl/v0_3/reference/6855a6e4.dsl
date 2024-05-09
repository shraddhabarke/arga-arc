(do
    (rule
        (vars this var)
        (filter
            (and
                (color_equals (color_of this) X)     
                (and
                    (neighbor_of obj)
                    (color_equals (color_of var) R)
                )
            )
        )
        (apply
            (mirror (mirror_axis var))
        )
    )
)