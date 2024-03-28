(do
    (rule
        (filter
            (filter_by_size 3)
        )
        (apply
            (move_node up)
            (update_color B)
        )
    )
)