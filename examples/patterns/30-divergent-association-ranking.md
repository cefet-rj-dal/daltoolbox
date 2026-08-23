About the method

- `pat_rule_filter_interest`: filters rules by classical interestingness
  measures such as `lift`, `Kulc`, and `IR`.
- `pat_rule_filter_dara`: filters rules by Divergent Association Ranking
  Analysis (DARA).
- `pat_rule_filter_none`: keeps all discovered rules and makes the
  no-filter option explicit.
- `pat_filter_rules`: applies an interesting-rule filter to a rule table
  or an `arules` rule object.

Didactic goal: separate pattern discovery from rule-interest filtering.
`pat_apriori`, ECLAT, and other miners discover candidate patterns; rule
filters keep the most interesting association rules according to either
classical measures or DARA.

Environment setup.

    source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
    # install.packages(c("daltoolbox", "arules"))

    library(daltoolbox)

    ## 
    ## Attaching package: 'daltoolbox'

    ## The following object is masked from 'package:base':
    ## 
    ##     transform

Load categorical data.

    data("AdultUCI", package = "arules")

    data <- as.data.frame(AdultUCI)
    data <- data[, c(
      "workclass",
      "education",
      "marital-status",
      "occupation",
      "income"
    )]

Mine target association rules and keep them in tabular form. This is the
candidate rule set before interestingness filtering.

    rules <- pat_dara_rules(
      data,
      rhs = "income=small",
      supp = 0.05,
      conf = 0.6,
      minlen = 2,
      maxlen = 4,
      remove_redundant = FALSE
    )

    head(rules[, c("lhs", "rhs", "support", "confidence", "lift", "count")])

    ##                                                   lhs          rhs    support
    ## 1                            occupation=Other-service income=small 0.06465747
    ## 2                             marital-status=Divorced income=small 0.08148724
    ## 3                        marital-status=Never-married income=small 0.20867286
    ## 4          workclass=Private,occupation=Other-service income=small 0.05405184
    ## 5           workclass=Private,marital-status=Divorced income=small 0.05781909
    ## 6 education=Some-college,marital-status=Never-married income=small 0.05896564
    ##   confidence     lift count
    ## 1  0.6414788 1.267440  3158
    ## 2  0.6000302 1.185545  3980
    ## 3  0.6323758 1.249454 10192
    ## 4  0.6507271 1.285713  2640
    ## 5  0.6005955 1.186662  2824
    ## 6  0.6467550 1.277864  2880

Keep all rules with the explicit no-filter option.

    none_rules <- pat_filter_rules(rules, pat_rule_filter_none())
    nrow(none_rules)

    ## [1] 9

Filter rules with classical interestingness measures. This keeps rules
with high lift, acceptable Kulczynski, and low imbalance ratio.

    interest_filter <- pat_rule_filter_interest(
      lift_min = 1.2,
      kulc_min = 0.38,
      ir_max = 0.8
    )

    interest_rules <- pat_filter_rules(rules, interest_filter)
    interest_rules[, c("lhs", "lift", "kulc", "ir")]

    ##                                                   lhs     lift      kulc
    ## 1                            occupation=Other-service 1.267440 0.3846148
    ## 3                        marital-status=Never-married 1.249454 0.5223367
    ## 6 education=Some-college,marital-status=Never-married 1.277864 0.3816299
    ## 7      education=HS-grad,marital-status=Never-married 1.287173 0.3872826
    ## 8      workclass=Private,marital-status=Never-married 1.268146 0.4798582
    ##          ir
    ## 1 0.7474797
    ## 3 0.2807310
    ## 6 0.7708135
    ## 7 0.7609306
    ## 8 0.4286892

Filter rules with DARA. This scores each rule antecedent using
divergence counters computed from the target subset.

    dara_filter <- pat_rule_filter_dara(
      data,
      rhs_attr = "income",
      rhs_value = "small",
      min_score = 1
    )

    dara_rules <- pat_filter_rules(rules, dara_filter)
    dara_rules[, c("lhs", "dara_score")]

    ##                                          lhs dara_score
    ## 1                   occupation=Other-service          2
    ## 2                    marital-status=Divorced          1
    ## 4 workclass=Private,occupation=Other-service          2
    ## 5  workclass=Private,marital-status=Divorced          1

DARA can also be inspected as a ranking to explain why rules were
retained.

    ranking <- attr(dara_rules, "dara_ranking")
    ranking[["marital-status"]]

    ##                   value dataset rules  c
    ## 1    Married-civ-spouse    8284     0 -1
    ## 2     Married-AF-spouse      13     0  0
    ## 3 Married-spouse-absent     384     0  0
    ## 4         Never-married   10192     5  0
    ## 5             Separated     959     0  0
    ## 6               Widowed     908     0  0
    ## 7              Divorced    3980     2  1

Use the DARA filter directly inside `pat_apriori`.

    pm <- pat_apriori(
      target = "rules",
      supp = 0.05,
      conf = 0.6,
      minlen = 2,
      maxlen = 4,
      rhs = "income=small",
      rule_filter = dara_filter,
      control = list(verbose = FALSE)
    )

    pm <- fit(pm, data)
    filtered_rules <- discover(pm, data)
    length(filtered_rules)

    ## [1] 4

    attr(filtered_rules, "dara_score")

    ## [1] 2 1 2 1

What to observe

- Interestingness filters are post-processing components: the miner
  generates candidate rules, and the filter decides which rules remain.
- `pat_rule_filter_none()` is the explicit default when the user wants
  discovery without post-mining rule filtering.
- `lift`, `Kulc`, and `IR` evaluate a rule directly from support
  relationships.
- DARA evaluates whether antecedent values become more prominent in the
  rule set than in the target subset.
- DARA scores are attached to filtered rules through the `dara_score`
  column or attribute, and the explanatory ranking is available as
  `dara_ranking`.
