(do
    (rule
        (with (this p q)) 
        (filter
            (and
                (== (color this) O) 
                (not (== (color p) O))
                (== (color p) (color q))
                (not (== p q))
                (can_see this p dir=horizontal)
                (can_see this q dir=horizontal)
            ) 
        )
        (apply
            (recolor this (color p)) 
        )
    )
)