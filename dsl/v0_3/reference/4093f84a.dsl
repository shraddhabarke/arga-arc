(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_size this 1)
                (filter_by_color v X)
            )
        )
        (apply
            (move_node_max v)
            (update_color X)
        )
    )
)