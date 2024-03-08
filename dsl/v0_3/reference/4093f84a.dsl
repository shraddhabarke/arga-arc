(do
    (rule
        (with this v)
        (filter
            (and
                (is_size this 1)
                (is_color v X)
            )
        )
        (apply
            (move_node_max v)
            (update_color X)
        )
    )
)