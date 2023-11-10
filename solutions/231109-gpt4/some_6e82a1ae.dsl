(do
    (
        (
            (filter_by_size min)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_size max)
            (
                (update_color C3)
            )
        )
        (
            (filter_by_color C5)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (and
                (filter_by_size 5)
                (filter_by_neighbor_size 0)
            )
            (
                (update_color C1)
            )
        )
        (
            (and
                (filter_by_size 6)
                (filter_by_neighbor_size 2)
            )
            (
                (update_color C2)
            )
        )
        (
            (and
                (filter_by_size 7)
                (filter_by_neighbor_size 2)
            )
            (
                (update_color C3)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C5)
            (
                (update_color C2)
            )
        )
        (
            (filter_by_color C5)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_size 1)
            (
                (update_color C3)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C5)
            (
                (update_color C1) (move_node U)
            )
        )
        (
            (filter_by_color C1)
            (
                (update_color C2) (move_node U)
            )
        )
        (
            (filter_by_color C2)
            (
                (update_color C3) (move_node U)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C5)
            (
                (update_color C3)
            )
        )
        (
            (filter_by_color C3)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_color C1)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (filter_by_degree 1)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_degree 3)
            (
                (update_color C2)
            )
        )
        (
            (filter_by_degree 2)
            (
                (update_color C3)
            )
        )
    )
)
(do
    (
        (
            (filter_by_size min)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_size 4)
            (
                (update_color C2)
            )
        )
        (
            (filter_by_size 9)
            (
                (update_color C3)
            )
        )
    )
)
(do
    (
        (
            (filter_by_size 1)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_size 5)
            (
                (update_color C2)
            )
        )
        (
            (filter_by_neighbor_size 5)
            (
                (update_color C3)
            )
        )
    )
)
(do
    (
        (
            (filter_by_neighbor_size 0)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_size 2)
            (
                (update_color C2)
            )
        )
        (
            (filter_by_size 3)
            (
                (update_color C3)
            )
        )
    )
)
(do
    (
        (
            (filter_by_size 1)
            (
                (update_color C1)
            )
        )
        (
            (filter_by_size 2)
            (
                (update_color C2)
            )
        )
        (
            (filter_by_size 3)
            (
                (update_color C3)
            )
        )
    )
)