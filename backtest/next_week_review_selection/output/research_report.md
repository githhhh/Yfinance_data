# Next Week Review Selection Research

Status: retrospective_pre_registered_replay

## Core hypothesis
B0 keeps every ACTIONABLE active signal. R1 keeps B0 unchanged and only adds
Near-Buy-Point UNCONFIRMED / BELOW_TRIGGER candidates with >=1 positive quality evidence.
Missing/False evidence is neutral; EXTENDED is exploratory only.

## B0 vs primary R1
variant,weeks,picks,avg_watchlist_size,median_watchlist_size,p95_watchlist_size,opportunity_recall_1w,non_actionable_opportunity_recall_1w,opportunities_per_review,winner_return_recall_1w,winner_mfe_recall_1w,big_winner_recall_1w,big_loser_inclusion_1w,big_loser_exclusion_1w,big_loser_density_1w,severe_loser_exposure_1w,winner_return_top10pct_recall_1w,loser_return_bottom10pct_inclusion_1w,median_return_1w_pct,median_mfe_1w_pct,median_mae_1w_pct,winner_return_recall_2w,winner_mfe_recall_2w,big_winner_recall_2w,big_loser_inclusion_2w,big_loser_exclusion_2w,big_loser_density_2w,severe_loser_exposure_2w,winner_return_top10pct_recall_2w,loser_return_bottom10pct_inclusion_2w,median_return_2w_pct,median_mfe_2w_pct,median_mae_2w_pct,winner_return_recall_3w,winner_mfe_recall_3w,big_winner_recall_3w,big_loser_inclusion_3w,big_loser_exclusion_3w,big_loser_density_3w,severe_loser_exposure_3w,winner_return_top10pct_recall_3w,loser_return_bottom10pct_inclusion_3w,median_return_3w_pct,median_mfe_3w_pct,median_mae_3w_pct,winner_return_recall_4w,winner_mfe_recall_4w,big_winner_recall_4w,big_loser_inclusion_4w,big_loser_exclusion_4w,big_loser_density_4w,severe_loser_exposure_4w,winner_return_top10pct_recall_4w,loser_return_bottom10pct_inclusion_4w,median_return_4w_pct,median_mfe_4w_pct,median_mae_4w_pct,big_winner_recall_mean_2_4w,big_loser_exclusion_mean_2_4w,big_loser_density_mean_2_4w,severe_loser_exposure_mean_2_4w,opportunity_recall_1w_delta_vs_b0,non_actionable_opportunity_recall_1w_delta_vs_b0,big_winner_recall_mean_2_4w_delta_vs_b0,big_loser_exclusion_mean_2_4w_delta_vs_b0,big_loser_density_mean_2_4w_delta_vs_b0,severe_loser_exposure_mean_2_4w_delta_vs_b0,avg_watchlist_size_delta_vs_b0
B0_ACTIONABLE_ONLY,19,554,29.157894736842106,24.0,68.39999999999989,0.34983498349834985,0.0,1.0,0.2631578947368421,0.22105263157894736,0.24166666666666667,0.23529411764705882,0.7647058823529411,0.12264150943396226,0.11084905660377359,0.2777777777777778,0.22839506172839505,0.2384641716181357,3.011704849390906,-3.000372174558769,0.22105263157894736,0.16842105263157894,0.21487603305785125,0.22054380664652568,0.7794561933534743,0.1721698113207547,0.16037735849056603,0.24691358024691357,0.25308641975308643,0.959842861230642,4.632310557985663,-3.8927507822685947,0.18947368421052632,0.21052631578947367,0.20869565217391303,0.20595533498759305,0.7940446650124069,0.1957547169811321,0.18867924528301888,0.21604938271604937,0.2037037037037037,1.665180410704148,5.602454367034804,-4.139408992741456,0.17894736842105263,0.17894736842105263,0.1864406779661017,0.21961620469083157,0.7803837953091685,0.2429245283018868,0.22877358490566038,0.2236024844720497,0.2422360248447205,2.137701989741214,6.536871201969907,-4.524602698338476,0.20333745439928866,0.7846282178916831,0.20361635220125787,0.19261006289308177,,,,,,,
R1_NEAR_BUY_POINT_PLUS_EVIDENCE,19,1472,77.47368421052632,71.0,157.19999999999993,0.8646864686468647,0.7918781725888325,0.9184925503943909,0.6210526315789474,0.47368421052631576,0.5416666666666666,0.6380090497737556,0.3619909502262444,0.12357581069237511,0.1016652059596845,0.6790123456790124,0.6358024691358025,0.21540611838832824,2.8994444097115046,-2.831083673984147,0.5578947368421052,0.42105263157894735,0.5206611570247934,0.6253776435045317,0.37462235649546827,0.18141980718667836,0.1691498685363716,0.6172839506172839,0.6481481481481481,1.271257159721384,4.55384474534255,-3.6983866159327916,0.5473684210526316,0.5052631578947369,0.5304347826086957,0.6129032258064516,0.3870967741935484,0.21647677475898336,0.20946538124452235,0.6111111111111112,0.5925925925925926,1.8216943299783628,5.936756768953999,-4.095801204615479,0.5473684210526316,0.5052631578947369,0.5423728813559322,0.6375266524520256,0.36247334754797444,0.26205083260297984,0.25153374233128833,0.5900621118012422,0.6024844720496895,2.2403272670728835,7.022906335396928,-4.553941203486311,0.5311562736631404,0.37473082607899705,0.21998247151621383,0.2100496640373941,0.5148514851485149,0.7918781725888325,0.3278188192638518,-0.4098973918126861,0.016366119314955963,0.01743960114431234,48.315789473684205


## Walk-forward train-selected champions
_No data_

## OOS rule stability
_No data_

## Retrospective champion candidate
- status: NO_STABLE_NEXT_WEEK_REVIEW_RULE
- rule: n/a

## EXTENDED exploratory lane
extension_bucket,rows,evaluable_1w,retest_to_buy_zone_1w_rate,median_return_1w_pct,median_return_4w_pct,median_mae_4w_pct
+5_to_10,260,179,0.48044692737430167,0.03375319787890518,0.9427568547955345,-6.607987597189624
+10_to_15,103,64,0.25,-0.016193781615864156,-1.5351119052717466,-9.354776189182623
>+15,113,65,0.07692307692307693,0.24647780198752134,0.9850142372804904,-11.12273431212699


## Guardrails
- 1W measures whether the weekend list captured a next-week review opportunity.
- 2W/3W/4W measure follow-through quality, winner capture and loser exposure.
- Big winners/losers are defined within each snapshot's full active-signal universe.
- C Rank, ATR and new technical indicators are excluded.
- No production Skill/Futu/Dashboard change is authorized by this retrospective study.
