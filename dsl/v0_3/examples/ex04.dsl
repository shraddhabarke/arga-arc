(do
    (rule
        (with this v)
        (filter
            (and
                (filter_by_size this 1)
                (filter_by_color v X)
                (is_neighbor this v)
            )
        )
        (apply
            (update_color R)
            (move_node_max D)
        )
    )
    (rule
        (with this v)
        (filter
            (and
                (filter_by_size this 1)
                (filter_by_color v Y)
                (is_neighbor this v)
            )
        )
        (apply
            (update_color B)
            (move_node_max U)
        )
    )
)