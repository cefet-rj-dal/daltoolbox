# Transformações (Preprocessamento)

Lista de exemplos de transformação/preprocessamento com breve descrição e link para cada `.Rmd`.

- [categorical_mapping](categorical_mapping.md): Mapeamento categórico (one‑hot); converte coluna categórica em variáveis binárias.
- [curvature_maximum](curvature_maximum.md): Máxima curvatura (segunda derivada em spline); ponto de trade‑off para curvas decrescentes.
- [curvature_minimum](curvature_minimum.md): Mínima curvatura (segunda derivada em spline); ponto de trade‑off para curvas crescentes.
- [dal_pca](dal_pca.md): PCA; projeta variáveis em componentes ortogonais ordenadas por variância explicada.
- [dal_smoothing_clustering](dal_smoothing_clustering.md): Suavização por agrupamento; discretiza por bins definidos via clustering.
- [dal_smoothing_frequency](dal_smoothing_frequency.md): Suavização por frequência (quantis); bins com contagens semelhantes.
- [dal_smoothing_interval](dal_smoothing_interval.md): Suavização por intervalos regulares (largura igual); sumariza contínuas em faixas.
- [na_removal](na_removal.md): Remoção de NAs; usa `na.omit` para eliminar instâncias com faltantes.
- [normalization_minmax](normalization_minmax.md): Min‑Max; reescala atributos para [0,1]; útil para algoritmos sensíveis à escala.
- [normalization_zscore](normalization_zscore.md): Z‑score; padroniza para média 0 e desvio 1 (ou `nmean`/`nsd`).
- [outliers_boxplot](outliers_boxplot.md): Outliers por boxplot (Q1−1,5×IQR, Q3+1,5×IQR); pode removê‑los.
- [outliers_gaussian](outliers_gaussian.md): Outliers gaussianos; valores além de média ± 3 desvios assumindo normalidade aproximada.
- [sample_random](sample_random.md): Amostragem aleatória; divide treino/teste e folds por sorteio.
- [sample_stratified](sample_stratified.md): Amostragem estratificada; preserva proporção do alvo por categoria em treino/teste e folds.

