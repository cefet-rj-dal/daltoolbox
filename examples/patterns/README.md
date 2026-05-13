# Pattern Mining Examples

This section introduces pattern mining as a family of descriptive learners that discover structure directly from co-occurrence data. The didactic emphasis is different from classification and regression: there is no target to predict, but there is still a stable Experiment Line built around `fit()`, `discover()`, and `evaluate()`.

For learners, the key question here is: how do we configure the search space, filter the discovered patterns, and summarize their quality in a reproducible way?

## Association Rules

Start with association rules, where the user can constrain left-hand side, right-hand side, and quality thresholds through object properties.

- [01-apriori-rules.md](/examples/patterns/01-apriori-rules.md) - `pat_apriori`: discovers association rules and evaluates them with support, confidence, lift, and retained-pattern summaries.

## Frequent Itemsets

Move next to itemset discovery, where the focus is on which items frequently appear together rather than directional rules.

- [02-eclat-itemsets.md](/examples/patterns/02-eclat-itemsets.md) - `pat_eclat`: mines frequent itemsets and shows how include/exclude filters and quality summaries guide interpretation.

## Sequential Patterns

Finish with sequences, where order matters and the interpretation shifts from co-occurrence to recurring temporal/event structure.

- [03-cspade-sequences.md](/examples/patterns/03-cspade-sequences.md) - `pat_cspade`: mines sequential patterns and evaluates them through support, pattern count, and sequence length summaries.
