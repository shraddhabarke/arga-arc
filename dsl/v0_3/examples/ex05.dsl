(do
    (rule
        (with this x)
        (filter
            (and
                (filter_by_size x 4)
                (is_neighbor this x)
            )
        )
        (apply
            (move_node_max x)
        )
    )
)