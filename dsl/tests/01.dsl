(do
    ;; Rule 1
    (rule
        ;; Filter
        (and
            (or 
                (filter_by_color B)
                (filter_by_color R)
            )
            (not (filter_by_size 4))
        )
        (apply
            ;; Transformation 1
            (move_node_max U)
            ;; Transformation 2
            (update_color X)
        )
    )
    ;; Rule 2
    (rule
        ;; Filter
        (not (filter_by_color G))
        (apply
            ;; Transformation 1
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