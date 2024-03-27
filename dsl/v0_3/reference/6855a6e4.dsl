(do
    (rule
        (filter
            (and
                (filter_by_color X)     
                (varand
                    (is_direct_neighbor)
                    (filter_by_color R)
                )
            )
        )
        (apply
            (mirror var_mirror)
        )
    )
)