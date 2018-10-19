for i in {1..15}; do
    in_file="out_JEW_km3_v4_nueCC_${i}.evt.root"
    echo $in_file
    out_name="out_JTP_km3_v4_nueCC_${i}.evt.root"
    echo $out_name_
    JTriggerProcessor -a km3net_115.detx -@ trigger_pars_arca_l0.txt -f $in_file -o $out_name
done

for i in {1..15}; do
    for typ in "eCC" "eNC" "muCC" "K40"; do
        for a in "" "a"; do
            in_file="out_JPT_km3_v4_${a}nu${typ}_${i}.evt.root"
            out_name="out_JTP_km3_v4_${a}nu${typ}_${i}.evt.root"

            mv $in_file $out_name
        done
    done
done

