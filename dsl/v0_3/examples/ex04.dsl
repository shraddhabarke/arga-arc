(do
    (rule
        (filter
            (and
                (neighbor_of obj)
                (size_equals obj_size 4)
            )
        )
        (apply
            (move_node_max vardirection)
        )
    )
)