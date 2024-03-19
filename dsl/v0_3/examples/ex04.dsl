(do
    (rule
        (filter
            (varand
                (is_diagonal_neighbor)
                (filter_by_size 4)
            )
        )
        (apply
            (move_node_max var)
        )
    )
)