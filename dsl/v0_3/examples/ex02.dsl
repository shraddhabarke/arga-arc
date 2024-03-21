(do
    (rule
        (filter
            (and
                (filter_by_color X)
                (filter_by_size min)
            )
        )
        (apply
            (update_color Y)
        )
    )
    (rule
        (filter
            (and
                (filter_by_color X)
                (filter_by_size min)
            )
        )
        (apply
            (update_color F)
        )
    )
    (rule
        (filter
            (and
                (filter_by_color X)
                (filter_by_size min)
            )
        )
        (apply
            (update_color A)
        )
    )
)