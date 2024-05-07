(do
    (rule
        (filter
            (and
                (color_equals obj_color X)     
                (and
                    (neighbor_of obj)
                    (color_equals obj_color R)
                )
            )
        )
        (apply
            (mirror varmirror)
        )
    )
)