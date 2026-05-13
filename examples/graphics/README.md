# Graphics Examples

This section groups plots by analytical question instead of by helper name. The numbering now uses semantic gaps so the reader can immediately see the transition from comparison charts to distribution views, relationship plots, time-oriented visuals, and export-focused examples.

For learning purposes, each block below can be read as a compact visual toolkit for a specific communication goal.

## Category Comparison

These charts are useful when the main goal is to compare categories, summarize aggregates, or show composition.

- [01-comparison-bar.md](/examples/graphics/01-comparison-bar.md) - bar chart for aggregated values by category.
- [02-comparison-bar-with-error.md](/examples/graphics/02-comparison-bar-with-error.md) - bar chart with uncertainty or variability bands through `geom_errorbar()`.
- [03-comparison-grouped-bar.md](/examples/graphics/03-comparison-grouped-bar.md) - grouped bars for side-by-side category comparisons.
- [04-comparison-stacked-bar.md](/examples/graphics/04-comparison-stacked-bar.md) - stacked bars for composition within each category.
- [05-comparison-lollipop.md](/examples/graphics/05-comparison-lollipop.md) - lollipop chart as a lighter alternative to bars.
- [06-proportion-pie.md](/examples/graphics/06-proportion-pie.md) - pie chart for simple proportion summaries.
- [07-profile-radar.md](/examples/graphics/07-profile-radar.md) - radar chart for multivariate profile comparison.

## Distribution Analysis

These plots are useful when the reader wants to inspect spread, skewness, outliers, or the overall shape of a numeric variable.

- [10-distribution-histogram.md](/examples/graphics/10-distribution-histogram.md) - histogram for binned frequency inspection.
- [11-distribution-density.md](/examples/graphics/11-distribution-density.md) - kernel density curve for smoothed distribution shapes.
- [12-distribution-boxplot.md](/examples/graphics/12-distribution-boxplot.md) - boxplot for quartiles, spread, and outliers.
- [13-distribution-boxplot-by-class.md](/examples/graphics/13-distribution-boxplot-by-class.md) - grouped boxplots for class-wise distribution comparison.
- [14-distribution-density-by-class.md](/examples/graphics/14-distribution-density-by-class.md) - grouped density curves for overlap and separation inspection.

## Relationships Between Variables

These plots help investigate association, overlap, and separation among variables or groups.

- [20-relationship-scatter.md](/examples/graphics/20-relationship-scatter.md) - scatter plot for two numeric variables.
- [21-relationship-points.md](/examples/graphics/21-relationship-points.md) - point plot for discrete or index-based observations without connecting lines.
- [22-relationship-correlation.md](/examples/graphics/22-relationship-correlation.md) - correlation heatmap for compact association inspection.
- [23-relationship-pair.md](/examples/graphics/23-relationship-pair.md) - scatter matrix for several numeric variables.
- [24-relationship-pair-advanced.md](/examples/graphics/24-relationship-pair-advanced.md) - scatter matrix with explicit class palette control.
- [25-relationship-parallel.md](/examples/graphics/25-relationship-parallel.md) - parallel coordinates for multivariate profile comparison.
- [26-relationship-pixel.md](/examples/graphics/26-relationship-pixel.md) - pixel-oriented matrix visualization.
- [27-relationship-dendrogram.md](/examples/graphics/27-relationship-dendrogram.md) - dendrogram view of hierarchical structure.

## Time-Oriented Views

These examples are useful for ordered or temporal data, where sequence matters as much as value.

- [30-time-series-lines.md](/examples/graphics/30-time-series-lines.md) - multi-line view for ordered series.
- [31-time-series-basic.md](/examples/graphics/31-time-series-basic.md) - simple exploratory time-series display.
- [32-time-series-forecast.md](/examples/graphics/32-time-series-forecast.md) - observed series with fit and forecast horizon.

## Export and Delivery

These final examples focus on saving figures for reports, slides, and publications.

- [40-export-save-pdf.md](/examples/graphics/40-export-save-pdf.md) - export charts to PDF with controlled dimensions.
- [41-export-save-jpg.md](/examples/graphics/41-export-save-jpg.md) - export charts to raster formats such as JPG.
