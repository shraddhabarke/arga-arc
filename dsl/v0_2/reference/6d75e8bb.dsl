(do
    (rule
        (with (this p q)) 
        (filter
            (and
                (== (color this) O) 
                (== (color p) C)
                (== (color q) C)
                (can_see this q dir=horizontal)
                (can_see this q dir=vertical)
            ) 
        )
        (apply
            (recolor this R) 
        )
    )
)