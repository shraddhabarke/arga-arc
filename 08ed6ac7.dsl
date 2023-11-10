(
    ;; Rule 1
    (
        (not (filter_by_color C0))
        (
            (update_color MOST)
            (move_node UP)
            (add_border C2)
            (extend_node DOWN)
            (extend_node UP True)
        )
    )
    ;; Rule 2
    (
        (filter_by_neighbor_color C2)
        (
            (move_node_max DOWN)
            (fill_rectangle C9 True)
            (hollow_rectangle C3)
            (flip VERTICAL)
        )
    )
    ;; Rule 3
    (
        (filter_by_size S12)
        (
            (move_node_max DOWN_LEFT)
            (rotate_node 90)
            (mirror (1 2))
        )
    )
    ;; Rule 4
    (
        (and (filter_by_degree D1) (and (filter_by_neighbor_size MAX) (filter_by_size S12)))
        (
            (mirror (1 null))
        )
    )
)