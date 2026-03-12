# Graphics Examples

This section now groups plots by analytical question instead of by helper name. That makes the collection easier to navigate when the reader is thinking in terms of what needs to be communicated: comparison, distribution, relationships, time series, or export.

For learning purposes, each group below can be read as a small visual toolkit for a particular type of analytical statement.

## Category Comparison

These charts are useful when the main goal is to compare categories, summarize aggregates, or show composition.

- [01-comparison-bar.md](/examples/graphics/01-comparison-bar.md) - Bar: compares aggregated values by categories. Useful for means, counts, and totals by group.
- [02-comparison-bar-with-error.md](/examples/graphics/02-comparison-bar-with-error.md) - Bars with error bars: add uncertainty/variability (e.g., standard deviation) to bars using `geom_errorbar()`.
- [03-comparison-grouped-bar.md](/examples/graphics/03-comparison-grouped-bar.md) - Grouped bars: compares multiple measures per category, displaying side-by-side bars.
- [04-comparison-stacked-bar.md](/examples/graphics/04-comparison-stacked-bar.md) - Stacked bars: shows each category's composition by summing groups; useful for cumulative proportions.
- [05-comparison-lollipop.md](/examples/graphics/05-comparison-lollipop.md) - Lollipop: alternative to bars, emphasizes values with a marker and a stem; lighter visual for comparisons.
- [06-proportion-pie.md](/examples/graphics/06-proportion-pie.md) - Pie: represents proportions of a total. Use sparingly and with few categories when angles are easy to compare.
- [07-profile-radar.md](/examples/graphics/07-profile-radar.md) - Radar: displays multiple numeric variables on radial axes from a common origin; useful for comparative profiles.

## Distribution Analysis

These plots are useful when the reader wants to inspect spread, skewness, outliers, or the overall shape of a numeric variable.

- [08-distribution-histogram.md](/examples/graphics/08-distribution-histogram.md) - Histogram: distributes observations into bins along the x-axis; useful to visualize frequency and skewness.
- [09-distribution-density.md](/examples/graphics/09-distribution-density.md) - Density (kernel density): smoothed version of the histogram for continuous variables; highlights distribution shapes.
- [10-distribution-boxplot.md](/examples/graphics/10-distribution-boxplot.md) - Boxplot: summarizes distribution via quartiles and highlights outliers; comparable across groups.

## Relationships Between Variables

These plots help investigate association, overlap, and separation among variables or groups.

- [11-relationship-scatter.md](/examples/graphics/11-relationship-scatter.md) - Scatter: assesses the relationship between two numeric variables, with optional coloring by group/category.
- [12-relationship-points.md](/examples/graphics/12-relationship-points.md) - Points: similar to series, but without connecting lines; good to highlight discrete observations.

## Time-Oriented Views

These examples are useful for ordered or temporal data, where sequence matters as much as value.

- [13-time-series-lines.md](/examples/graphics/13-time-series-lines.md) - Time series (lines): points connected by segments; highlights trend and seasonality over time/ordered axis.
- [14-time-series-basic.md](/examples/graphics/14-time-series-basic.md) - Simple time series: exploratory visualization of a temporal vector with ordered x-axis and y values.
- [15-time-series-forecast.md](/examples/graphics/15-time-series-forecast.md) - Time series with fit and forecast: shows observed values, model fit, and forecast horizon for visual comparison.

## Export and Delivery

These final examples focus on saving figures so they can be used in reports, slides, or publications.

- [16-export-save-pdf.md](/examples/graphics/16-export-save-pdf.md) - Chart saving: example of exporting with `ggsave()` to PDF, controlling dimensions and units.
- [17-export-save-jpg.md](/examples/graphics/17-export-save-jpg.md) - Chart saving: example of exporting with `jpeg()`/`dev.off()` and `ggsave()` to image files.
