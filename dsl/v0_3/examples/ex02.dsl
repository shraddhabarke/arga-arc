(do
    (rule
        (vars (this))
        (filter
            (and
                (color_equals (color_of this) X)
                (size_equals (size_of this) min)
            )
        )
        (apply
            (update_color Y)
        )
    )
    (rule
        (vars (this))
        (filter
            (and
                (color_equals (color_of this) X)
                (size_equals (size_of this) min)
            )
        )
        (apply
            (update_color F)
        )
    )
    (rule
        (vars (this))
        (filter
            (and
                (color_equals (color_of this) X)
                (size_equals (size_of this) min)
            )
        )
        (apply
            (update_color A)
        )
    )
)