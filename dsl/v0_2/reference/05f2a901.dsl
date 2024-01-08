(do
    (rule
        (with (this c)) 
        (filter
            (and
                (== (color this) R) 
                (== (color c) C) 
            )
        )
        (apply
            (push this (to this c)) 
        )
    )
)