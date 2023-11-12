(do
    (
        (
            (filter_by_neighbor_color C2)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
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
            (filter_by_neighbor_color C2)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C1)
            (
                (fill_rectangle C2 True)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C0)
            (
                (update_color C2)
            )
        )
    )
)
;; after adding comments to the grammar
(do
    (
        (
            (filter_by_neighbor_color C2)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C2)
            (
                (update_color C1)
            )
        )
    )
)
(do
    (
        (
            (filter_by_neighbor_color C2)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (filter_by_neighbor_color C2)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C2)
            (
                (extend_node U)
                (extend_node D)
                (extend_node L)
                (extend_node R)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C2)
            (
                (move_node U)
                (fill_rectangle C2 True)
            )
        )
    )
)
(do
    (
        (
            (filter_by_neighbor_color C2)
            (
                (update_color C2)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C2)
            (
                (update_color C1)
            )
        )
    )
)
(do
    (
        (
            (filter_by_color C2)
            (
                (move_node U)
            )
        )
        (
            (filter_by_color C2)
            (
                (add_border C1)
            )
        )
    )
)