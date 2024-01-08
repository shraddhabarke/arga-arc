(do
    (rule
        (with (this s)) 
        (filter
            (and 
                (not (== (size this) 1))
                (== (size s) 1)
            )
        )
        (apply
            (recolor this (color s)) 
        )
    )
)