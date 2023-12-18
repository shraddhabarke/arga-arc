(do
    (rule
        (with (this v))
        (filter
            (and
                (or
                    (== (color v) B)
                    (== (color v) R)
                )
                (can_see this v)
            )
        )
        (apply
            (push this (to this v) n=0 overlap=T trail=no)
        )
    )
)