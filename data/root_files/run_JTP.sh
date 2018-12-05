# Run JTP on the test and val data set
for i in {13..15}; do
    for typ in "eCC" "eNC" "muCC" "K40"; do
        for a in "" "a"; do

            in_file="root_files/out_JEW_km3_v4_${a}nu${typ}_${i}.evt.root"
            out_name="root_files/out_JTP_km3_v4_${a}nu${typ}_${i}.evt.root"

            echo $in_file
            echo $out_name

            JTriggerProcessor -a km3net_115.detx -@ trigger_pars_arca_l0.txt -f $in_file -o $out_name

        done
    done
done
