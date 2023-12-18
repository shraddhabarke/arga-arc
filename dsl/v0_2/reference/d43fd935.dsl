(do
    (rule
        (with
            (this g)
        )
        (filter
            (and
                (== (color g) G)
                (can_see this g)
            )
        )
        (apply
            (push this (to this g) n=0 overlap=F trail=keep)
        )
    )
)