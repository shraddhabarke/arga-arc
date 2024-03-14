(do
    (rule
        (with this)
        (filter
            (and
                (filter_by_color this X)
                (filter_by_size this MIN)
            )
        )
        (apply
            (update_color Y)
        )
    )
    (rule
        (with this)
        (filter
            (and
                (filter_by_color this X)
                (filter_by_size this MIN)
            )
        )
        (apply
            (update_color F)
        )
    )
    (rule
        (with this)
        (filter
            (and
                (filter_by_color this X)
                (filter_by_size this MIN)
            )
        )
        (apply
            (update_color A)
        )
    )
)