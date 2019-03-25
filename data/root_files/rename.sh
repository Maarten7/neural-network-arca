for i in {13..15}
do
    for typ in "eCC" "eNC" "muCC" "K40"
    do
        for a in "" "a"
        do
            infile="out_JTE_km3_v4_${a}nu${typ}_${i}.evt.root"
            outfile="out_JTE_${a}nu${typ}_${i}.root"
            mv $infile $outfile
        done
    done
done
