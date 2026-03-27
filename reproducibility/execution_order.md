# Execution Order

1. Obtain the required restricted data from the official CHARLS, KLoSA/KEIS, and Gateway access routes.
2. Run the CHARLS data-processing scripts.
3. Run the CHARLS stage scripts supporting the locked results.
4. Run the pragmatic-rule scripts.
5. Run the KLoSA processing/build scripts and then the KLoSA validation scripts.
6. If figure/table rebuild is needed, regenerate the aggregated result files locally and then use the internal build assets or master package rather than this lean public release package.
