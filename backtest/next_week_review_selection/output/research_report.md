# Next Week Review Selection Research

Status: retrospective_pre_registered_replay

## Core hypothesis
B0 keeps every ACTIONABLE active signal. Primary R1 keeps B0 unchanged and adds
Near-Buy-Point UNCONFIRMED / BELOW_TRIGGER candidates with >=1 independent
positive evidence family.

## Price-path coverage audit
rows,symbol_found_count,symbol_found_rate,complete_1w_count,complete_1w_rate,complete_2w_count,complete_2w_rate,complete_3w_count,complete_3w_rate,complete_4w_count,complete_4w_rate,state_COMPLETE_4W,state_MISSING_SYMBOL,state_SHORT_3W,state_SHORT_4W,state_NO_FORWARD_BARS
2738,1946,0.7107377647918188,1944,0.7100073046018992,1944,0.7100073046018992,1788,0.6530314097881665,1683,0.614682249817385,1683,792,156,105,2


## B0 vs Primary R1 — micro aggregation
variant,weeks,picks,avg_watchlist_size,median_watchlist_size,p95_watchlist_size,evaluable_picks_1w,opportunities_available_1w,opportunities_captured_1w,selection_coverage_1w,opportunity_recall_1w,non_actionable_opportunity_recall_1w,opportunities_per_review,big_winner_recall_1w,big_loser_inclusion_1w,winner_capture_lift_1w,loser_capture_lift_1w,median_return_1w_pct,median_mfe_1w_pct,median_mae_1w_pct,tradable_selection_coverage_1w,tradable_big_winner_recall_1w,tradable_big_loser_inclusion_1w,tradable_winner_capture_lift_1w,tradable_loser_capture_lift_1w,opp_severe_loser_exposure_1w,median_opp_return_1w_pct,median_opp_mfe_1w_pct,median_opp_mae_1w_pct,selection_coverage_2w,big_winner_recall_2w,big_loser_inclusion_2w,winner_capture_lift_2w,loser_capture_lift_2w,median_return_2w_pct,median_mfe_2w_pct,median_mae_2w_pct,tradable_selection_coverage_2w,tradable_big_winner_recall_2w,tradable_big_loser_inclusion_2w,tradable_winner_capture_lift_2w,tradable_loser_capture_lift_2w,opp_severe_loser_exposure_2w,median_opp_return_2w_pct,median_opp_mfe_2w_pct,median_opp_mae_2w_pct,selection_coverage_3w,big_winner_recall_3w,big_loser_inclusion_3w,winner_capture_lift_3w,loser_capture_lift_3w,median_return_3w_pct,median_mfe_3w_pct,median_mae_3w_pct,tradable_selection_coverage_3w,tradable_big_winner_recall_3w,tradable_big_loser_inclusion_3w,tradable_winner_capture_lift_3w,tradable_loser_capture_lift_3w,opp_severe_loser_exposure_3w,median_opp_return_3w_pct,median_opp_mfe_3w_pct,median_opp_mae_3w_pct,selection_coverage_4w,big_winner_recall_4w,big_loser_inclusion_4w,winner_capture_lift_4w,loser_capture_lift_4w,median_return_4w_pct,median_mfe_4w_pct,median_mae_4w_pct,tradable_selection_coverage_4w,tradable_big_winner_recall_4w,tradable_big_loser_inclusion_4w,tradable_winner_capture_lift_4w,tradable_loser_capture_lift_4w,opp_severe_loser_exposure_4w,median_opp_return_4w_pct,median_opp_mfe_4w_pct,median_opp_mae_4w_pct,big_winner_recall_mean_2_4w,snapshot_winner_capture_lift_mean_2_4w,snapshot_loser_capture_lift_mean_2_4w,tradable_big_winner_recall_mean_2_4w,tradable_winner_capture_lift_mean_2_4w,tradable_loser_capture_lift_mean_2_4w,opp_severe_loser_exposure_mean_2_4w,opportunity_recall_1w_delta_vs_b0,non_actionable_opportunity_recall_1w_delta_vs_b0,selection_coverage_1w_delta_vs_b0,big_winner_recall_mean_2_4w_delta_vs_b0,snapshot_winner_capture_lift_mean_2_4w_delta_vs_b0,snapshot_loser_capture_lift_mean_2_4w_delta_vs_b0,tradable_big_winner_recall_mean_2_4w_delta_vs_b0,tradable_winner_capture_lift_mean_2_4w_delta_vs_b0,tradable_loser_capture_lift_mean_2_4w_delta_vs_b0,opp_severe_loser_exposure_mean_2_4w_delta_vs_b0,avg_watchlist_size_delta_vs_b0,added_evaluable_reviews_vs_b0,incremental_opportunities_vs_b0,incremental_opportunities_per_added_review,attention_multiplier_vs_b0
B0_ACTIONABLE_ONLY,42,733,17.452380952380953,8.0,54.29999999999994,526,1532,526,0.2705761316872428,0.3433420365535248,0.0,1.0,0.22672064777327935,0.22965116279069767,0.8379181354966826,0.8487487841542134,0.21782368666852792,2.9486739905188575,-2.9029698920337466,0.3433420365535248,0.33476394849785407,0.2857142857142857,0.975015910834054,0.8321564367191743,0.08935361216730038,0.29571158173520296,3.0666663928358373,-2.808402468355531,0.2705761316872428,0.21721311475409835,0.21560574948665298,0.8027800286729414,0.7968395000039037,0.8144255445579951,4.519428993350494,-3.8072114348758843,0.34401569653368214,0.30962343096234307,0.3087818696883853,0.9000270455160125,0.89758075808658,0.16159695817490494,0.9048541431928236,4.64928601032113,-3.8154319757899957,0.2757270693512304,0.21212121212121213,0.21550094517958412,0.7693158768209478,0.7815734076695667,1.4161587974875767,5.5605458335892255,-4.145077720207258,0.3484098939929329,0.2831858407079646,0.29846938775510207,0.812795060044158,0.8566616301693092,0.1926977687626775,1.2699086443596075,5.818184939297755,-4.083487342792925,0.27213309566250743,0.2088888888888889,0.22107081174438686,0.7675982532751092,0.8123628300563386,2.137701989741214,6.507965108952085,-4.481090028543683,0.3423019431988042,0.2727272727272727,0.3116279069767442,0.7967447399761809,0.9103889509495278,0.24017467248908297,1.9826690234071576,6.830836291045506,-4.520159820274616,0.2127410719213998,0.7798980529229995,0.7969252459099363,0.28851218146586016,0.8365222818454505,0.8882104464018057,0.19815646647555515,,,,,,,,,,,,,,,
R1_NEAR_BUY_POINT_PLUS_EVIDENCE,42,1953,46.5,24.5,142.54999999999987,1443,1532,1325,0.7422839506172839,0.8648825065274152,0.794234592445328,0.9182259182259183,0.6396761133603239,0.6656976744186046,0.8617674042775259,0.896823478218827,0.22385575247678702,2.8356400969479667,-2.7120316833136537,0.8648825065274152,0.7982832618025751,0.8262548262548263,0.9229961940238076,0.9553376557150142,0.08754716981132075,0.5846550926366545,3.0795257898714956,-2.5459685228951257,0.7422839506172839,0.6188524590163934,0.6468172484599589,0.8337139156811287,0.871387893975163,0.9885564039828099,4.398861094417739,-3.71496108340168,0.8652714192282538,0.799163179916318,0.8243626062322946,0.9235982631081256,0.95272140962145,0.1655328798185941,1.1412762369195306,4.725103436495526,-3.4142522057913682,0.7466442953020134,0.6277056277056277,0.6389413988657845,0.8407023687922565,0.85575072746968,1.5988509184788668,5.7550164828552575,-4.1864427469544445,0.8657243816254417,0.7743362831858407,0.8010204081632653,0.8944374209860935,0.9252603082049146,0.20408163265306123,1.6775011000499163,6.000394339131221,-3.951401050788095,0.7468805704099821,0.64,0.6545768566493955,0.8568973747016707,0.876414359380217,2.292091451526379,7.043103730513889,-4.595704580799154,0.8609865470852018,0.7818181818181819,0.8186046511627907,0.9080492424242425,0.9507751937984495,0.2526041666666667,2.3306640867024164,7.252084985418672,-4.374786354015087,0.628852695574007,0.8437712197250186,0.8678509936083533,0.7851058816401135,0.9086949755061539,0.9429189705416047,0.20740622637944064,0.5215404699738904,0.794234592445328,0.4717078189300411,0.41611162365260723,0.06387316680201904,0.07092574769841697,0.49659370017425336,0.07217269366070334,0.054708524139798986,0.00924975990388549,29.047619047619047,917.0,799.0,0.871319520174482,2.6643929058663027


## Weekly macro aggregation
variant,weeks,macro_mean_opportunity_recall_1w,macro_median_opportunity_recall_1w,macro_mean_selection_coverage_1w,macro_median_selection_coverage_1w,macro_mean_opportunities_per_review,macro_median_opportunities_per_review,macro_mean_tradable_big_winner_recall_mean_2_4w,macro_median_tradable_big_winner_recall_mean_2_4w,macro_mean_tradable_winner_capture_lift_mean_2_4w,macro_median_tradable_winner_capture_lift_mean_2_4w,macro_mean_tradable_loser_capture_lift_mean_2_4w,macro_median_tradable_loser_capture_lift_mean_2_4w,macro_mean_opp_severe_loser_exposure_mean_2_4w,macro_median_opp_severe_loser_exposure_mean_2_4w,macro_mean_avg_watchlist_size,macro_median_avg_watchlist_size
B0_ACTIONABLE_ONLY,42,0.30488948729380777,0.2847368421052632,0.24413357333044602,0.2264957264957265,1.0,1.0,0.2904982363315697,0.20317460317460317,0.9800679608047334,1.0,0.9269181892952307,0.9415695415695415,0.20834123273417307,0.16129032258064516,17.452380952380953,8.0
R1_NEAR_BUY_POINT_PLUS_EVIDENCE,42,0.8343890259568677,0.856907894736842,0.7460663008747351,0.75,0.8824276093380212,0.9128787878787878,0.7811318972033258,0.788888888888889,0.9371195143124584,1.0,0.9859474625101239,1.0,0.26245398976652506,0.24074074074074073,46.5,24.5


## Paired moving-block bootstrap
metric,weeks,block_size,observed_macro_delta,bootstrap_ci_2_5,bootstrap_ci_97_5,prob_delta_gt_0
opportunity_recall_1w,42,4,0.5294995386630599,0.46970352401772447,0.6108153144385314,1.0
avg_watchlist_size,42,4,29.047619047619047,14.950000000000001,44.431547619047606,1.0
tradable_big_winner_recall_2w,42,4,0.4835600907029479,0.41637188208616777,0.5632100340136055,1.0
tradable_winner_capture_lift_2w,42,4,-0.13715816015785307,-0.2972046923309551,-0.022747508633018304,0.006
tradable_loser_capture_lift_2w,42,4,0.05999446106927529,-0.08829667101858914,0.2400974312093693,0.796
opp_severe_loser_exposure_2w,42,4,0.050078725201980555,0.00022585436127444544,0.11453881173885574,0.975
tradable_big_winner_recall_3w,42,4,0.4768002322880372,0.4105946677451468,0.5589441609977324,1.0
tradable_winner_capture_lift_3w,42,4,-0.10129447835375402,-0.21645089633626796,0.0077008379612925,0.038
tradable_loser_capture_lift_3w,42,4,0.1343819081064618,-0.009799965562191957,0.26539313135537757,0.964
opp_severe_loser_exposure_3w,42,4,0.07088749348059069,0.0013062740077962106,0.16114128643965447,0.9775
tradable_big_winner_recall_4w,42,4,0.4978472222222222,0.43723516744096014,0.5823015873015873,1.0
tradable_winner_capture_lift_4w,42,4,0.042722068217609536,-0.10069359658063398,0.1736170817628179,0.7085
tradable_loser_capture_lift_4w,42,4,-0.03199990032670282,-0.15671594978060313,0.11315580611326413,0.3615
opp_severe_loser_exposure_4w,42,4,0.06348326241014576,-0.0084294053663723,0.1432929833070644,0.955


## Walk-forward train-selected champions
fold,asof_cutoff,train_start,train_end,test_start,test_end,train_snapshot_weeks,resolved_1w_train_weeks,resolved_2w_train_weeks,resolved_3w_train_weeks,resolved_4w_train_weeks,champion_rule,champion_rule_json
1,2026-03-13,2025-10-10,2026-03-06,2026-03-13,2026-04-02,20,20,19,18,16,NO_STABLE_CANDIDATE,{}
2,2026-04-10,2025-10-10,2026-04-02,2026-04-10,2026-05-01,24,24,22,21,20,NO_STABLE_CANDIDATE,{}
3,2026-05-08,2025-10-10,2026-05-01,2026-05-08,2026-05-29,28,28,27,26,25,NO_STABLE_CANDIDATE,{}
4,2026-06-05,2025-10-10,2026-05-29,2026-06-05,2026-06-26,32,32,30,29,28,NO_STABLE_CANDIDATE,{}
5,2026-07-02,2025-10-10,2026-06-26,2026-07-02,2026-07-24,36,35,34,33,32,NO_STABLE_CANDIDATE,{}
6,2026-07-31,2025-10-10,2026-07-24,2026-07-31,2026-08-07,40,40,39,38,37,NO_STABLE_CANDIDATE,{}


## OOS stability
evaluation_role,variant,folds,opportunity_positive_rate,tradable_winner_lift_nonnegative_rate,tradable_loser_lift_nonworse_rate,mean_opportunity_delta,mean_tradable_winner_lift_delta,mean_tradable_loser_lift_delta,mean_incremental_opportunity_efficiency,mean_attention_delta,rule_complexity,stability_floor
PRIMARY_R1,R1_NEAR_BUY_POINT_PLUS_EVIDENCE,6,1.0,0.5,0.5,0.5111152594906989,0.06340269574696039,-0.025564887234469962,0.8562595100786629,47.25,0,0.5


## Retrospective champion candidate
- status: NO_STABLE_NEXT_WEEK_REVIEW_RULE
- rule: n/a

## EXTENDED exploratory lane
extension_bucket,rows,evaluable_1w,retest_to_buy_zone_1w_rate,median_snapshot_return_4w_pct,median_snapshot_mae_4w_pct,median_post_retest_return_4w_pct,median_post_retest_mae_4w_pct
+5_to_10,260,179,0.48044692737430167,0.9427568547955345,-6.607987597189624,2.898032239058801,-5.244570502166162
+10_to_15,103,64,0.25,-1.5351119052717466,-9.354776189182623,1.3888875091518038,-10.637265182832323
>+15,113,65,0.07692307692307693,0.9850142372804904,-11.12273431212699,7.712580770297727,-3.4000323313381378


## Guardrails
- Price-path missingness is explicitly audited by week/status/setup/source/ticker.
- Walk-forward training is horizon-aware and as-of censored by true label end date.
- Snapshot and opportunity clocks remain separate.
- Winner recall is interpreted together with Selection Coverage and capture lift.
- Rule evolution remains two-stage: structural grid then evidence-family ablation.
- C Rank, ATR and new technical indicators are excluded.
- No production Skill/Futu/Dashboard change is authorized by this retrospective study.
