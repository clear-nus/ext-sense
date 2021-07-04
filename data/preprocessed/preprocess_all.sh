preprocess_nuskin () {
    for freq in 5 50 100 250 500 1000 2000 3000 4000
    do
        python preprocess.py $1 $2 $freq
    done
}

preprocess_biotac () {
    for freq in 5 50 100 250 500 1000 2000 2200
    do
        python preprocess.py $1 $2 $freq
    done
}


preprocess_nuskin nuskin_tool_20 $1
preprocess_nuskin nuskin_tool_30 $1
preprocess_nuskin nuskin_tool_50 $1
preprocess_nuskin nuskin_handover_rod $1
preprocess_nuskin nuskin_handover_box $1
preprocess_nuskin nuskin_handover_plate $1
preprocess_nuskin nuskin_food_apple $1
preprocess_nuskin nuskin_food_banana $1
preprocess_nuskin nuskin_food_empty $1
preprocess_nuskin nuskin_food_pepper $1
preprocess_nuskin nuskin_food_tofu $1
preprocess_nuskin nuskin_food_water $1
preprocess_nuskin nuskin_food_watermelon $1

preprocess_biotac biotac_tool_20 $1
preprocess_biotac biotac_tool_30 $1
preprocess_biotac biotac_tool_50 $1
preprocess_biotac biotac_handover_rod $1
preprocess_biotac biotac_handover_box $1
preprocess_biotac biotac_handover_plate $1
preprocess_biotac biotac_food_apple $1
preprocess_biotac biotac_food_banana $1
preprocess_biotac biotac_food_empty $1
preprocess_biotac biotac_food_pepper $1
preprocess_biotac biotac_food_tofu $1
preprocess_biotac biotac_food_water $1
preprocess_biotac biotac_food_watermelon $1
