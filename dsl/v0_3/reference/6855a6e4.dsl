(do
    (rule
        (vars (this other))
        (filter
            (and
                (color_equals (color_of this) X)     
                (and
                    (neighbor_of this other)
                    (color_equals (color_of other) R)
                )
            )
        )
        (apply
            (mirror (mirror_axis_of other))
        )
    )
)