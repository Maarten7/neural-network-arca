for i in {1..15}; do
    in_file="out_JEW_km3_v4_nuK40_${i}.evt.root"
    echo $in_file
    out_name="out_JTP_km3_v3_nuK40_${i}.root"
    echo $out_name_
    JTriggerProcessor -a km3net_115.detx -@ trigger_pars_arca_l0.txt -f $in_file -o $out_name

    in_file="out_JEW_km3_v4_anuK40_${i}.evt.root"
    echo $in_file
    out_name="out_JTP_km3_v3_anuK40_${i}.root"
    echo $out_name
    JTriggerProcessor -a km3net_115.detx -@ trigger_pars_arca_l0.txt -f $in_file -o $out_name
done

mv *JTP* JTP_files/
