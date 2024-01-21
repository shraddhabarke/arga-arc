(do
    (rule
        (with 
            (this y)
        )
        (filter 
            (and
                (== (color this) X)
                (== (size y) 1)
                (can_see x y)
            )
        )
        (apply
            (updatecolor this (color y))
        )
    )
)