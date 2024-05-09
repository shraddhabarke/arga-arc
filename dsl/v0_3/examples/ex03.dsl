(do
    (rule
        (vars (this))
        (filter)
        (apply
            (update_color R)
        )
    )
    (rule
        (vars (this))
        (filter
            (size_equals (size_of this) 1)
        )
        (apply
            (update_color R)
        )
    )
)