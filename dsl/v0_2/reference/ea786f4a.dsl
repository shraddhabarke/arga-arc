(do
    (rule
        (with (this)) 
        (filter
            (== (color this) O)
        )
        (apply
            (push this UL n=0 overlap=T trail=reset) 
            (push this UR n=0 overlap=T trail=reset) 
            (push this DL n=0 overlap=T trail=reset) 
            (push this DR n=0 overlap=T trail=reset) 
        )
    )
)