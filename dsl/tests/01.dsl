(do
    (rule
        (and
            (or 
                (filter_by_color B)
                (filter_by_color R)
            )
            (not (filter_by_size 4))
        )
        (apply
            (move_node_max U)
            (update_color X)
        )
    )
    (rule
        (filter_by_size MAX)
        (apply
            (extend_node D overlap=T)
        )
    )
)

(do
    (rule
        ()
        (apply
            (update_color Y)
        )
    )
)

(do
    (rule
        (and
            (filter_by_color X)
            (filter_by_size MIN)
        )
        (apply
            (update_color R)
        )
    )
    (rule
        (and
            (filter_by_color X)
            (filter_by_size MIN)
        )
        (apply
            (update_color G)
        )
    )
    (rule
        (and
            (filter_by_color X)
            (filter_by_size MIN)
        )
        (apply
            (update_color B)
        )
    )
)

(do
    (rule
        ()
        (apply
            (update_color R)
        )
    )
    (rule
        (filter_by_size 1)
        (apply
            (update_color B)
        )
    )
)