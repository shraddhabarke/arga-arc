(do
    (rule
        (vars (this))
        (filter
            (size_equals (size_of this) 3)
        )
        (apply
            (move_node up)
            (update_color B)
        )
    )
)