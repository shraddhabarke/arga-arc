(do
    (rule
        (with (this b)) 
        (filter
            (and
                (== (color b) O) 
                (can_see this b)
            )
        )
        (apply
            (recolor this G) 
        )
    )
)