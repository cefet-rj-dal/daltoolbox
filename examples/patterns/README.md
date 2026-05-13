# Pattern Mining Examples

This section introduces pattern mining as a family of descriptive learners that discover structure directly from co-occurrence data. The numbering now leaves semantic gaps because association rules, itemsets, and sequential patterns are related but conceptually distinct mining tasks.

The didactic question here is different from classification and regression: how do we configure the search space, constrain the discovered patterns, and summarize their quality in a reproducible way?

## Association Rules

Start here if you want directional patterns with left-hand side and right-hand side interpretation.

- [01-apriori-rules.md](/examples/patterns/01-apriori-rules.md) - `pat_apriori`: discovers association rules and evaluates them with support, confidence, lift, and retained-pattern summaries.

## Frequent Itemsets

Move next to itemset discovery, where the focus is on what appears together rather than on directional implication.

- [10-eclat-itemsets.md](/examples/patterns/10-eclat-itemsets.md) - `pat_eclat`: mines frequent itemsets and shows how include/exclude filters guide interpretation.

## Sequential Patterns

Finish with sequence mining, where order matters and the interpretation shifts from co-occurrence to recurring event structure.

- [20-cspade-sequences.md](/examples/patterns/20-cspade-sequences.md) - `pat_cspade`: mines sequential patterns and evaluates them through support and sequence summaries.
