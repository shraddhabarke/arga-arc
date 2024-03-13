(do
    (rule
        (filter 
            (and
                (== (color this) grey)
                (== (size y) 1)
                (neighbor this y)
            )
        )
    )
)